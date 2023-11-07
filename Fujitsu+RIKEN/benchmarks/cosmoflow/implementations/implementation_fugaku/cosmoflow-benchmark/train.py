"""
Main training script for the CosmoFlow Keras benchmark
"""

# System imports
print("# start python script")
import os
import argparse
import logging
import pickle
from types import SimpleNamespace
import re
import random
import subprocess

# External imports
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from mpi4py import MPI

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if MPI.COMM_WORLD.Get_rank() == 0:
    tf.compat.v1.logging.set_verbosity(logging.INFO)
else:
    tf.compat.v1.logging.set_verbosity(logging.ERROR)
#import horovod.tensorflow.keras as hvd
from mlperf_logging import mllog
from tensorflow.python.client import timeline
import time


# Local imports
from data import get_datasets
from models import get_model
# Fix for loading Lambda layer checkpoints
from models.layers import *
from utils.optimizers import get_optimizer, get_lr_schedule
from utils.callbacks import TimingCallback, MLPerfLoggingCallback, ProfilingCallback, TerminateOnBaseline, TerminateOnDivergence
from utils.device import configure_session
from utils.argparse import ReadYaml
from utils.checkpoints import reload_last_checkpoint, reload_checkpoint
from utils.mlperf_logging import configure_mllogger, log_submission_info
from utils.hooks import CustomInMemoryEvaluatorHook, EpochLogHook

# Stupid workaround until absl logging fix, see:
# https://github.com/tensorflow/tensorflow/issues/26691
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

# Keras auto mixed precision
#from tensorflow.keras.mixed_precision import experimental as mixed_precision

tf.compat.v1.disable_eager_execution()
import mesh_tensorflow as mtf

import uuid
#dummy_output_dir='/local/cosmoflow-{}'.format(uuid.uuid4().hex)
dummy_output_dir='/tmp/cosmoflow-{}'.format(uuid.uuid4().hex)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser('train.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/cosmo.yaml')
    add_arg('--output-dir', help='Override output directory')

    # Override data settings
    add_arg('--data-dir', help='Override the path to input files')
    add_arg('--n-train', type=int, help='Override number of training samples')
    add_arg('--n-valid', type=int, help='Override number of validation samples')
    add_arg('--batch-size', type=int, help='Override the batch size')
    add_arg('--n-epochs', type=int, help='Override number of epochs')
    add_arg('--apply-log', type=int, choices=[0, 1], help='Apply log transform to data')
    add_arg('--stage-dir', help='Local directory to stage data to before training')

    # Hyperparameter settings
    add_arg('--conv-size', type=int, help='CNN size parameter')
    add_arg('--fc1-size', type=int, help='Fully-connected size parameter 1')
    add_arg('--fc2-size', type=int, help='Fully-connected size parameter 2')
    add_arg('--hidden-activation', help='Override hidden activation function')
    add_arg('--dropout', type=float, help='Override dropout')
    add_arg('--optimizer', help='Override optimizer type')
    add_arg('--lr', type=float, help='Override learning rate')

    # Other settings
    add_arg('-d', '--distributed', action='store_true')
    add_arg('--rank-gpu', action='store_true',
            help='Use GPU based on local rank')
    add_arg('--resume', type=str,
            help='Resume from last checkpoint')
    add_arg('--print-fom', action='store_true',
            help='Print parsable figure of merit')
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--mixed_precision', action='store_true')

    add_arg('--timeline', type=str, help='Output path to json format timeline', default=None)
    add_arg('--prestaged', action='store_true', help='data is already staged to stage-dir')
    add_arg('--seed', type=int, help='Random number seed (not works !)', default=-1)
    add_arg('--num_mpiar_tensors', type=int, help='Number of mpi-allreduce tensors (default 999)', default=999)
    add_arg('--th-inter', type=int, help='Number of inter-threads for TensorFlow (default 1)', default=1)
    add_arg('--th-intra', type=int, help='Number of intra-threads for TensorFlow (default 48)', default=48)
    add_arg('--target-mae', type=float, help='Stop training when validation mae reachs this')
    add_arg('--do-augmentation', action='store_true')
    add_arg('--validation-batch-size', type=int, help='Batch size for validation')
    add_arg('--train-staging-dup-factor', type=int, help='N times more samples are staged for training')
    add_arg('--mesh-shape', type=str, help='Mesh Shape')

    add_arg('--ready-dir', help='directory for synchonization with other instances')
    add_arg('--num-instances', type=int, help='The number of instances', default=1)
    add_arg('--instance-num', type=int, help='Instance num', default=0)
    add_arg('--prof-step', type=int, help='Steps of TF-Profile', default=0)

    return parser.parse_args()

def init_workers(distributed=False, model_parallel_size=(1,1)):
    if distributed:
        assert len(model_parallel_size) == 2, 'model_parallel_size must be [X, Y]'
        mtf.mesh_init(axis_d_block=model_parallel_size[0], axis_h_block=model_parallel_size[1])

        class Dist:
            def __init__(self, model_parallel_size):
                self._comm = MPI.COMM_WORLD
                self._rank = self._comm.Get_rank()
                self._size = self._comm.Get_size()
                if 'MPI_LOCALRANKID' in os.environ:
                    self._local_rank = int(os.environ['MPI_LOCALRANKID'])
                elif 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
                    self._local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
                else:
                    self._local_rank = 0

                if 'MPI_LOCALNRANKS' in os.environ:
                    self._local_size = int(os.environ['MPI_LOCALNRANKS'])
                elif 'OMPI_COMM_WORLD_LOCAL_SIZE' in os.environ:
                    self._local_size = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
                else:
                    self._local_size = 1

                self._model_parallel_size = model_parallel_size # list[int]
                n_model_para = model_parallel_size[0] * model_parallel_size[1]

                assert self.size % n_model_para == 0, '# of processes is not multiple of # of model parallel processes'
                self._data_parallel_size = self.size // n_model_para # int
                self._data_parallel_rank = self.rank // n_model_para
                model_parallel_flat_rank = mtf.get_model_rank()
                self._model_parallel_rank = (model_parallel_flat_rank // self.model_parallel_size[1], model_parallel_flat_rank % self.model_parallel_size[1])
                assert self.data_parallel_rank < self.data_parallel_size
                assert self.model_parallel_rank[0] < self.model_parallel_size[0]
                assert self.model_parallel_rank[1] < self.model_parallel_size[1]

                #assert self._size % n_model_para == 0, 'model parallel proces must be in same node' # FIXME process group?
                self._data_parallel_local_size = max(self.local_size // n_model_para, 1)
                self._data_parallel_local_rank = self.local_rank // n_model_para

            @property
            def rank(self): return self._rank
            @property
            def size(self): return self._size
            @property
            def local_rank(self): return self._local_rank
            @property
            def local_size(self): return self._local_size
            @property
            def model_parallel_size(self): return self._model_parallel_size
            @property
            def data_parallel_size(self): return self._data_parallel_size
            @property
            def model_parallel_rank(self): return self._model_parallel_rank
            @property
            def data_parallel_rank(self): return self._data_parallel_rank
            @property
            def data_parallel_local_size(self): return self._data_parallel_local_size
            @property
            def data_parallel_local_rank(self): return self._data_parallel_local_rank

            def barrier(self):
                self._comm.Barrier()

        return Dist(model_parallel_size)
    else:
        return SimpleNamespace(rank=0, size=1, local_rank=0, local_size=1,
                               model_parallel_size=[1,1],
                               data_parallel_size=1,
                               model_parallel_rank=[0,0],
                               data_parallel_rank=0,
                               data_parallel_local_size=1,
                               data_parallel_local_rank=0,
        )

def config_logging(verbose):
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format=log_format)

def load_config(args):
    """Reads the YAML config file and returns a config dictionary"""
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Expand paths
    output_dir = config['output_dir'] if args.output_dir is None else args.output_dir
    config['output_dir'] = os.path.expandvars(output_dir)

    # Override data config from command line
    if args.data_dir is not None:
        config['data']['data_dir'] = args.data_dir
    if args.n_train is not None:
        config['data']['n_train'] = args.n_train
    if args.n_valid is not None:
        config['data']['n_valid'] = args.n_valid
    if args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size
    if args.n_epochs is not None:
        config['data']['n_epochs'] = args.n_epochs
    if args.apply_log is not None:
        config['data']['apply_log'] = bool(args.apply_log)
    if args.stage_dir is not None:
        config['data']['stage_dir'] = args.stage_dir
    if args.prestaged:
        config['data']['prestaged'] = args.prestaged
    config['data']['seed'] = args.seed
    if args.do_augmentation:
        config['data']['do_augmentation'] = args.do_augmentation
    if args.validation_batch_size:
        config['data']['validation_batch_size'] = args.validation_batch_size
    config['data']['train_staging_dup_factor'] = args.train_staging_dup_factor or 1

    # Hyperparameters
    if args.conv_size is not None:
        config['model']['conv_size'] = args.conv_size
    if args.fc1_size is not None:
        config['model']['fc1_size'] = args.fc1_size
    if args.fc2_size is not None:
        config['model']['fc2_size'] = args.fc2_size
    #if args.l2 is not None:
    #    config['model']['l2'] = args.l2
    if args.hidden_activation is not None:
        config['model']['hidden_activation'] = args.hidden_activation
    if args.dropout is not None:
        config['model']['dropout'] = args.dropout
    if args.optimizer is not None:
        config['optimizer']['name'] = args.optimizer
    if args.lr is not None:
        config['optimizer']['lr'] = args.lr
    if args.mesh_shape is not None:
        config['model']['mesh_shape'] = [int(i) for i in args.mesh_shape.split(',')]

    return config

def save_config(config):
    output_dir = config['output_dir']
    config_file = os.path.join(output_dir, 'config.pkl')
    logging.info('Writing config via pickle to %s', config_file)
    with open(config_file, 'wb') as f:
        pickle.dump(config, f)

def load_history(output_dir):
    return pd.read_csv(os.path.join(output_dir, 'history.csv'))

def print_training_summary(output_dir, print_fom):
    history = load_history(output_dir)
    if 'val_loss' in history.keys():
        best = history.val_loss.idxmin()
        logging.info('Best result:')
        for key in history.keys():
            logging.info('  %s: %g', key, history[key].loc[best])
        # Figure of merit printing for HPO parsing
        if print_fom:
            print('FoM:', history['val_loss'].loc[best])

def main():
    """Main function"""

    # Initialization
    args = parse_args()

    # Set random seed
    if args.seed != -1:
        np.random.seed(seed=args.seed)
        random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = '0'

    config = load_config(args)
    mesh_shape = config['model'].get('mesh_shape')
    dist = init_workers(args.distributed, mesh_shape)

    os.makedirs(config['output_dir'], exist_ok=True)
    config_logging(verbose=args.verbose)

    # Start MLPerf logging
    mllogger = configure_mllogger(config['output_dir'])
    if dist.rank == 0:
        log_submission_info(dist, **config.get('mlperf', {}))

        mllogger.start(key=mllog.constants.CACHE_CLEAR)

    dist.barrier()
    if dist.rank == 0:
        mllogger.start(key=mllog.constants.INIT_START)
        mllogger.event(key='seed', value = args.seed)
        # Scale logging for mlperf hpc metrics
        mllogger.event(key='number_of_ranks', value=dist.size)
        mllogger.event(key='number_of_nodes', value=(dist.size//dist.local_size))
        mllogger.event(key='accelerators_per_node', value=0) # Fugaku has no accelerators.

    logging.info('Initialized rank %i size %i local_rank %i local_size %i',
                 dist.rank, dist.size, dist.local_rank, dist.local_size)
    if dist.rank == 0:
        logging.info('Configuration: %s', config)

    # Device and session configuration
    gpu = dist.local_rank if args.rank_gpu else None
    if gpu is not None:
        logging.info('Taking gpu %i', gpu)
    print("#intra_threads, inter_threads", args.th_intra, args.th_inter)
    configure_session(intra_threads=args.th_intra, inter_threads=args.th_inter, seed=args.seed, **config.get('device', {}))

    data_config = config['data']
    if dist.rank == 0:
        mllogger.event(key=mllog.constants.TRAIN_SAMPLES, value=data_config['n_train'])
        mllogger.event(key=mllog.constants.EVAL_SAMPLES, value=data_config['n_valid'])
        mllogger.event(key=mllog.constants.GLOBAL_BATCH_SIZE, value=data_config['batch_size'] * dist.data_parallel_size)

    # Construct or reload the model
    if dist.rank == 0:
        logging.info('Building the model')
        mllogger.event(key='dropout', value=config['model']['dropout'])
    train_config = config['train']
    initial_epoch = 0
    checkpoint_format = os.path.join(config['output_dir'], 'checkpoint-{epoch:03d}.h5')

#    if args.timeline:
#        run_options = tf.compat.v1.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#    else:
#        run_options = tf.compat.v1.RunOptions()
#    run_metadata = tf.compat.v1.RunMetadata()

    if args.resume:
        if os.path.isdir(args.resume):
            resume_checkpoint_format = os.path.join(args.resume, 'checkpoint-{epoch:03d}.h5')
            # Reload model from last checkpoint
            initial_epoch, model = reload_last_checkpoint(
                resume_checkpoint_format, data_config['n_epochs'],
                distributed=args.distributed)
        else:
            basename = os.path.basename(args.resume)
            m = re.fullmatch(r'checkpoint-([0-9][0-9][0-9])\.h5', basename)
            if m is None:
                raise Exception('Can not resume checkpoint file %s' % args.resume)
            initial_epoch = int(m.groups()[0])
            model = reload_checkpoint(args.resume, distributed=args.distributed)

    else:
        # if args.mixed_precision:
        #     # Set auto mixed precision policy
        #     policy = mixed_precision.Policy('mixed_float16')
        #     mixed_precision.set_policy(policy)


        # Configure the optimizer
        global_batch_size = data_config['batch_size'] * dist.data_parallel_size
        epoch_steps = data_config['n_train'] // global_batch_size
        lr_scheduler = get_lr_schedule(global_batch_size=global_batch_size, epoch_steps=epoch_steps, is_root = (dist.rank == 0), **config['lr_schedule'])

        opt_fn = get_optimizer(distributed=args.distributed, is_root = (dist.rank == 0),
                               **config['optimizer'])

        # Build a new model
        save_conf = {'enabled': dist.rank == 0,
                     'output_dir': config['output_dir'] if dist.rank == 0 else dummy_output_dir,
                     'save_steps': 100000000}
        model = get_model(**config['model'],
                          optimizer=opt_fn,
                          lr_scheduler=lr_scheduler,
                          loss=train_config['loss'], metrics=train_config['metrics'], batch_size=data_config['batch_size'], save_conf=save_conf,
                          num_mpiar_tensors=args.num_mpiar_tensors,
                          seed=args.seed if args.seed != -1 else None)
        session_config = tf.compat.v1.ConfigProto()

        cosmo_estimator = tf.estimator.Estimator(
            model_fn=model,
            model_dir=save_conf['output_dir'],
            config = tf.estimator.RunConfig(
                session_config=session_config,
                save_checkpoints_steps=100000000, # don't save checkpoint
                save_summary_steps=save_conf['save_steps'] if save_conf['enabled'] else 100000000,
                keep_checkpoint_max=0
            ))

#    if dist.rank == 0:
#        model.summary()

    # Save configuration to output directory
    if dist.rank == 0:
        config['n_ranks'] = dist.size
        save_config(config)

#    # Prepare the callbacks
#    if dist.rank == 0:
#        logging.info('Preparing callbacks')
#    callbacks = []
#    if args.distributed:
#
#        # Broadcast initial variable states from rank 0 to all processes.
#        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
#
#        # Average metrics across workers
#        callbacks.append(hvd.callbacks.MetricAverageCallback())
#
#    # Learning rate decay schedule
#    if 'lr_schedule' in config:
#        global_batch_size = data_config['batch_size'] * dist.size
#        callbacks.append(tf.keras.callbacks.LearningRateScheduler(
#            get_lr_schedule(global_batch_size=global_batch_size, is_root = (dist.rank == 0),
#                            **config['lr_schedule'])))
#
#    # Timing
#    timing_callback = TimingCallback()
#    callbacks.append(timing_callback)
#
#    # Terminate
#    if args.target_mae is not None:
#        terminate_callback = TerminateOnBaseline(baseline=args.target_mae)
#        callbacks.append(terminate_callback)

#    callbacks.append(TerminateOnDivergence(baseline=1.0))

#    if dist.rank == 0 and args.timeline:
#        profiling_callback = ProfilingCallback(run_metadata, args.timeline, max(0, datasets['n_train_steps']-16))
#        callbacks.append(profiling_callback)

    # Checkpointing and logging from rank 0 only
    # if dist.rank == 0:
    #     callbacks.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_format))
    #     callbacks.append(tf.keras.callbacks.CSVLogger(
    #         os.path.join(config['output_dir'], 'history.csv'), append=args.resume))
    #     #callbacks.append(tf.keras.callbacks.TensorBoard(
    #     #    os.path.join(config['output_dir'], 'tensorboard')))
    #     callbacks.append(MLPerfLoggingCallback())

    # # Early stopping
    # patience = config.get('early_stopping_patience', None)
    # if patience is not None:
    #     callbacks.append(tf.keras.callbacks.EarlyStopping(
    #         monitor='val_loss', min_delta=1e-5, patience=patience, verbose=1))

    # if dist.rank == 0:
    #     logging.debug('Callbacks: %s', callbacks)

    # Synchronize with other instances
    if args.distributed: dist.barrier()
    if args.num_instances > 1 and dist.rank == 0:
        assert args.ready_dir is not None

        # touch my file
        my_file_path = os.path.join(args.ready_dir, str(args.instance_num))
        print("touch", my_file_path, flush=True)
        with open(my_file_path, "w") as f:
            pass

        # wait for all instances
        check_sec = 5
        while True:
            num_ready = len(os.listdir(args.ready_dir))
            print("num_ready", num_ready, flush=True)
            if num_ready == args.num_instances:
                break
            time.sleep(check_sec)

        # get time that last touched
        ready_times = [os.stat( os.path.join(args.ready_dir, str(i)) ).st_mtime for i in range(args.num_instances)]
        max_ready_time = max(ready_times)

        # wait
        margin_sec = 30
        sleep_time = max_ready_time + 30 - time.time()
        if sleep_time > 0:
            print("wait for synchronize", sleep_time, flush=True)
            time.sleep(sleep_time)
        else:
            print("margin time is already past!")

    print(time.time(), "# wait for inter-job synchronize", flush=True)
    # inter-job synchronization
    jobsetname = os.getenv('JobSetName')
    if dist.rank == 0:
        if jobsetname is not None and jobsetname != "None":
            while not os.path.exists("/2ndfs/ra010011/cosmoflow/ready_flag/jobs/"+os.getenv('JobSetName')):
                time.sleep(1)

    # Run Start
    if args.distributed: dist.barrier()
    if dist.rank == 0:
        mllogger.end(key=mllog.constants.INIT_STOP)
        mllogger.start(key=mllog.constants.RUN_START)

    # Run staging
    if 'stage_dir' in data_config and 'prestaged' not in data_config:
        if dist.rank == 0:
            mllogger.start(key='staging_start')

        if args.num_instances >= 1:
            with open("/worktmp/staging_begin", "w") as f:
                pass

            while not os.path.exists("/worktmp/staging_end"):
                time.sleep(1)
        else:
            CompressType='none'
            cp_script = os.getenv('SCRIPT_DIR') + '/cpdata.sh'
            GroupSize = os.getenv('GroupSize')
            cp_script = cp_script + ' ' + data_config['data_dir'] + ' ' + data_config['stage_dir']
            cp_script = cp_script + ' ' + CompressType + ' ' + str(dist.size) + ' ' + GroupSize
            os.system(cp_script)

        data_config['prestaged'] = True

        if args.distributed: dist.barrier()
        if dist.rank == 0:
            mllogger.start(key='staging_stop')

    # Load the data
    if dist.rank == 0:
        logging.info('Loading data')
    datasets = get_datasets(dist=dist, **data_config)
    logging.debug('Datasets: %s', datasets)

    # Train the model
    if dist.rank == 0:
        logging.info('Beginning training')
    fit_verbose = 1 if (args.verbose and dist.rank==0) else 0

    if config['data']['name'].endswith('dali'):
        with tf.device('/gpu:0'):
            model.fit(datasets['train_dataset'],
                      steps_per_epoch=datasets['n_train_steps'],
                      epochs=data_config['n_epochs'],
                      validation_data=datasets['valid_dataset'],
                      validation_steps=datasets['n_valid_steps'],
                      callbacks=callbacks,
                      initial_epoch=initial_epoch,
                      verbose=fit_verbose)
    else:
        eval_hook = CustomInMemoryEvaluatorHook(
            cosmo_estimator,
            datasets['valid_dataset'],
            steps=datasets['n_valid_steps'],
            every_n_iter=datasets['n_train_steps'],
            initial_epoch=initial_epoch,
            target_mae=args.target_mae or 0)
        
        train_log_hook = tf.estimator.LoggingTensorHook(
            {"my_train_mae"}, every_n_iter=datasets['n_train_steps']
        )

        epoch_log_hook = EpochLogHook(train_epoch_steps=datasets['n_train_steps'], initial_epoch=initial_epoch)

##for profiler
        train_hooks=[train_log_hook, eval_hook, epoch_log_hook]

        if args.prof_step != 0:
          if dist.rank <= 1:
            profiler_hook = tf.estimator.ProfilerHook(
                save_steps=args.prof_step, save_secs=None,
                output_dir=config['output_dir'] + "/prof_" + str(dist.rank) + "/",
                show_dataflow=True,
                show_memory=True
                )
            train_hooks.append(profiler_hook)

        cosmo_estimator.train(input_fn=datasets['train_dataset'],
                              steps=datasets['n_train_steps'] * data_config['n_epochs'],
                              hooks=train_hooks)
        #eval_results = cosmo_estimator.evaluate(input_fn=datasets['valid_dataset'], steps=datasets['n_valid_steps'])
        #logging.info("\nEvaluation results:\n\t%s\n" % eval_results)

        #train_spec = tf.estimator.TrainSpec(input_fn=datasets['train_dataset'], max_steps=2000)
        #eval_spec = tf.estimator.EvalSpec(input_fn=datasets['valid_dataset'], steps=datasets['n_valid_steps'],
        #                                  start_delay_secs=3, throttle_secs=3)
        #tf.estimator.train_and_evaluate(cosmo_estimator, train_spec, eval_spec)

    # Stop MLPerf timer
    if args.distributed: dist.barrier()
    if dist.rank == 0:
        mllogger.end(key=mllog.constants.RUN_STOP, metadata={'status': 'success'})

    #if dist.rank == 0:
    #    print('Epoch times : {}'.format(timing_callback.times))

    # Print training summary
    #if dist.rank == 0:
    #    print_training_summary(config['output_dir'], args.print_fom)

    # Finalize
    #if dist.rank == 0:
    #    logging.info('All done!')
    logging.info('All done!')

if __name__ == '__main__':
    main()
