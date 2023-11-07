import argparse
import logging
import time

import data
import model
import utils
import random

import horovod.mxnet as hvd
import mxnet as mx
import numpy as np

from mxnet.contrib import amp
from typing import List, Tuple, Optional, Literal, Dict


@utils.ArgumentParser.register_extension()
def add_argument_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--config-file", type=str, 
                        help="Config file where basic configuration is stored")
    parser.add_argument("--log-prefix", type=str, help="Prefix fo logfile")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="Insert profiling method and ranges")
    parser.add_argument("--cuda-profiler-range", type=str, default="",
                        help="Cudaprofile only specific iterations")
    parser.add_argument("--seed", type=int, required=True, 
                        help="Seed used for training")
    exclusive_group = parser.add_mutually_exclusive_group()
    exclusive_group.add_argument("--instances", type=int, default=1, 
                                 help="Number of parallel instances for weak scaling")
    exclusive_group.add_argument("--spatial-span", type=int, default=1, 
                                 help="Number of GPU per which single spatial parallel is working")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="Path to the checkpoint that needs to be loaded at the beggining")
    parser.add_argument("--save-checkpoint", type=str, default=None,
                        help="Path where to store checkpoint after training")

@utils.ArgumentParser.register_extension("Training arguments")
def add_training_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of epochs to train")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--use-amp", action="store_true", default=False, 
                        help="Use mixed precision for training")
    group.add_argument("--use-fp16", action="store_true", default=False,
                        help="Use static loss scaling and fp16 training without AMP")

    parser.add_argument("--static-loss-scale", type=float, default=8192*2,
                        help="Static loss scaling used for static fp16 training")
    parser.add_argument("--grad-prediv-factor", type=float, default=1.0,
                        help="Gradient divider for all reduce step")

    parser.add_argument("--target-mae", type=float, default=0.0, 
                        help="Stop training when validation reach specific target value.")
    parser.add_argument("--initial-lr", type=float, default=0.1,
                        help="Initial learning rate used.")
    parser.add_argument("--base-lr", type=float, default=0.1,
                        help="Learning rate after warmup steps.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Training momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Weight decay during training. If zero, dropout will be used.")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout factor")
    parser.add_argument("--warmup-epochs", type=int, default=4,
                        help="Number of epochs for linear warmup")
    parser.add_argument("--lr-scheduler-epochs", type=int, nargs=2, default=[32, 64],
                        help="Epochs where to decrease learning rate")
    parser.add_argument("--lr-scheduler-decays", type=float, nargs=2, default=[0.25, 0.125],
                        help="Factors how much decrease learning rate on decay epochs")


class LRScheduler():
    def __init__(self, initial_lr: float, 
                 peak_lr: float,
                 warmup_epochs: int,
                 steps_per_epoch: int,
                 drop_steps: List[Tuple[int, int]] = [(32, 0.25), (64, 0.125)]):
        self.base_lr = peak_lr
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.steps_per_epoch = steps_per_epoch
        self.drop_steps = drop_steps
        self.drops_idx = 0

        self.current_lr = self.base_lr
        self.last_epoch = -1
        self.last_iter = 0
        self.first_iter = 0

        utils.logger.event(key=utils.logger.constants.OPT_BASE_LR, 
                           value=self.base_lr)
        utils.logger.event(key=utils.logger.constants.OPT_LR_WARMUP_EPOCHS, 
                           value=self.warmup_epochs)
        utils.logger.event(key=utils.logger.constants.OPT_LR_WARMUP_FACTOR, 
                           value=int(self.base_lr / self.initial_lr))
        utils.logger.event(key=utils.logger.constants.OPT_LR_DECAY_BOUNDARY_EPOCHS, 
                           value=[x[0] for x in self.drop_steps])
        utils.logger.event(key=utils.logger.constants.OPT_LR_DECAY_FACTOR, 
                           value=[x[1] for x in self.drop_steps])

    def start_baseline(self):
        self.first_iter = self.last_iter

    def __call__(self, iteration: int) -> float:
        current_epoch = (iteration - self.first_iter) // self.steps_per_epoch
        if current_epoch < self.warmup_epochs:
            lr = (self.base_lr - self.initial_lr) * current_epoch / self.warmup_epochs + self.initial_lr
        else:
            if (self.drops_idx < len(self.drop_steps) and 
                self.drop_steps[self.drops_idx][0] < current_epoch):
                self.current_lr = self.base_lr * self.drop_steps[self.drops_idx][1]
                self.drops_idx += 1
            lr = self.current_lr
        if current_epoch != self.last_epoch:
            self.last_epoch = current_epoch

        self.last_iter = iteration
        return lr



def main(args: argparse.Namespace):
    dist_desc = utils.DistributedEnvDesc.get_from_mpi(args.instances)

    utils.logger = utils.Logger(dist_desc.worker, args.log_prefix)
    utils.logger.register_dist_desc(dist_desc)

    if dist_desc.master:
        print(args)
    number_of_nodes = (dist_desc.size // dist_desc.local_size) * args.instances

    utils.logger.event(key=utils.logger.constants.CACHE_CLEAR)
    utils.logger.start(key=utils.logger.constants.INIT_START)

    utils.logger.event(key=utils.logger.constants.SUBMISSION_BENCHMARK, 
                       value="cosmoflow")
    utils.logger.event(key=utils.logger.constants.SUBMISSION_ORG, value="NVIDIA")
    utils.logger.event(key=utils.logger.constants.SUBMISSION_DIVISION, value="closed")
    utils.logger.event(key=utils.logger.constants.SUBMISSION_STATUS, value="onprem")
    utils.logger.event(key=utils.logger.constants.SUBMISSION_PLATFORM, 
                       value=f"{number_of_nodes}xNVIDIA DGX A100")

    utils.logger.event(key="number_of_nodes", value=dist_desc.size // dist_desc.local_size)
    utils.logger.event(key="accelerators_per_node", value=dist_desc.local_size)

    cuda_profile_opt = utils.parse_cuda_profile_argument(args.cuda_profiler_range) \
        if args.cuda_profiler_range != "" else None
    per_instance_seed = args.seed + (dist_desc.worker 
        if dist_desc.worker is not None else 0)

    mx.random.seed(per_instance_seed)
    np.random.seed(per_instance_seed)
    random.seed(per_instance_seed)

    model = initialize_model(dist_desc, use_amp=args.use_amp, 
                             use_fp16=args.use_fp16, 
                             use_wd=(args.weight_decay != 0.0),
                             data_layout=args.data_layout,
                             dropout=args.dropout,
                             batch_size=args.training_batch_size,
                             checkpoint=args.load_checkpoint,
                             spatial_span=args.spatial_span)
    utils.logger.event(key=utils.logger.constants.OPT_WEIGHT_DECAY,
                       value=args.weight_decay)
    utils.logger.event(key="dropout", 
                       value=args.dropout)

    iteration_builder, train_steps, val_steps = data.get_rec_iterators(args, dist_desc)

    lr_scheduler = LRScheduler(initial_lr=args.initial_lr, 
                               peak_lr=args.base_lr,
                               warmup_epochs=args.warmup_epochs,
                               steps_per_epoch=train_steps, 
                               drop_steps=[(k, v) for k, v in zip(args.lr_scheduler_epochs, 
                                                                  args.lr_scheduler_decays)])
    optimizer = mx.optimizer.SGD(learning_rate=None,
                                 momentum=args.momentum,
                                 lr_scheduler=lr_scheduler,
                                 wd=args.weight_decay,
                                 multi_precision=args.use_fp16,
                                 rescale_grad=1.0 if not args.use_fp16 else args.static_loss_scale)
    utils.logger.event(key=utils.logger.constants.OPT_NAME,
                       value=utils.logger.constants.SGD)

    if dist_desc.size == 1:
        trainer = mx.gluon.Trainer(model.collect_params(), optimizer,
                                   update_on_kvstore=False)
    else:
        network_parameters = model.collect_params()
        if network_parameters is not None:
            hvd.broadcast_parameters(network_parameters)

        gradient_predivide_factor = args.grad_prediv_factor
        trainer = hvd.DistributedTrainer(network_parameters, optimizer,
                                         gradient_predivide_factor=gradient_predivide_factor,
                                         num_groups=1)

    if args.use_amp:
        amp.init_trainer(trainer)

    lr_scheduler.start_baseline()
    dist_desc.MPI.COMM_WORLD.Barrier()
    utils.logger.stop(key=utils.logger.constants.INIT_STOP)
    utils.logger.start(key=utils.logger.constants.RUN_START)

    train_iter, val_iter = iteration_builder()

    for epoch in range(args.num_epochs):
        with utils.ProfilerSection(f"epoch_{epoch}", args.profile):
            cuda_profile_range = cuda_profile_opt[1:] if (cuda_profile_opt and 
                    epoch == cuda_profile_opt[0]) else None

            utils.logger.start(key=utils.logger.constants.EPOCH_START, 
                               metadata={'epoch_num': epoch + 1})

            fit_epoch(model, trainer, None, train_iter, args.training_batch_size, 
                      epoch_number=epoch, steps_per_epoch=train_steps,
                      spatial_span=args.spatial_span,
                      use_amp=args.use_amp, dist_desc=dist_desc, 
                      profile=args.profile, cuda_profile=cuda_profile_range,
                      use_fp16=args.use_fp16, static_loss_scale=args.static_loss_scale)
            validation_loss = validate(model, val_iter, args.validation_batch_size,
                                       epoch_number=epoch,
                                       use_fp16=args.use_fp16,
                                       dist_desc=dist_desc, profile=args.profile)

            
            utils.logger.stop(key=utils.logger.constants.EPOCH_STOP, 
                              metadata={'epoch_num': epoch + 1})
            utils.logger.event(key='eval_error', 
                               value=validation_loss,
                               metadata={'epoch_num': epoch + 1})

            if validation_loss <= args.target_mae:
                utils.logger.stop(key=utils.logger.constants.RUN_STOP, 
                                metadata={'status': 'success'})
                break
    else:
        utils.logger.stop(key=utils.logger.constants.RUN_STOP, 
                          metadata={'status': 'failure'})

    if (dist_desc.master and args.save_checkpoint is not None 
            and args.save_checkpoint != ""):
        model.save_parameters(args.save_checkpoint)
    
    # Lets sync all processes withing slurm job before exiting
    dist_desc.MPI.COMM_WORLD.Barrier()


def initialize_model(dist_desc: utils.DistributedEnvDesc, 
                     use_amp: bool = False, 
                     use_fp16: bool = False,
                     use_wd: bool = False,
                     data_layout: Literal["NCDHW", "NDHWC"] = "NDHWC", 
                     dropout: float = 0.5,
                     batch_size: int = 1,
                     checkpoint: Optional[str] = None,
                     spatial_span: int = 1):
    DTYPE = "float16" if use_fp16 else "float32"
    if use_amp:
        amp.init()

    network = model.CosmoflowWithLoss(dist_desc, conv_kernel=3, layout=data_layout, 
                                      use_wd=use_wd, dropout_rate=dropout, 
                                      spatial_span=spatial_span)
    if use_fp16:
        network.cast("float16")
    network.init(mx.gpu(dist_desc.local_rank), batch_size, 
                 use_amp, use_wd,
                 dist_desc, checkpoint, DTYPE)

    return network

def fit_epoch(model: mx.gluon.HybridBlock, trainer: mx.gluon.Trainer, 
              loss_fn, dataset_iterator, batch_size: int, epoch_number: int,
              steps_per_epoch: int, spatial_span: int = 1,
              use_amp: bool = False, log_every: int = 250,
              use_fp16: bool = False, static_loss_scale: int = 8192,
              *, 
              dist_desc: utils.DistributedEnvDesc,
              metric = mx.metric.MAE(),
              profile: bool = False,
              cuda_profile: Optional[Tuple[int, int]] = None):
    def _step(idx, data, label, perf_counter, prefetch_iter = None):
        with utils.ProfilerSection(f"iter_{idx}", profile=profile):
            with mx.autograd.record():
                if use_fp16:
                    data = mx.nd.cast(data, "float16")
                    output = model(data, label)
                    output = mx.nd.cast(output, "float32")
                else:
                    output = model(data, label)
                loss = output
                if prefetch_iter is not None:
                    try:
                        next_batch = next(prefetch_iter)
                    except StopIteration:
                        next_batch = None

                if use_amp:
                    with amp.scale_loss(loss, trainer) as scaled_loss:
                        mx.autograd.backward(scaled_loss)
                elif use_fp16:
                    mx.autograd.backward(loss * static_loss_scale)
                else:
                    mx.autograd.backward(loss)
            trainer.step(batch_size)
            metric.update(label, output)

            perf_counter.update_processed(batch_size * dist_desc.size // spatial_span)
            if (idx + 1) % log_every == 0 and dist_desc.master:
                logging.info(f"Batch {idx+1} out of {steps_per_epoch}. Current throughput = "
                            f"{perf_counter.throughput:.2f}/s")

            if prefetch_iter is not None:
                return next_batch
            
    performance_counter = utils.PerformanceCounter()
    metric.reset()

    overlap_prefetch = False

    if overlap_prefetch:
        iterator = iter(dataset_iterator)
        next_batch = next(iterator)

        item_processed = 0
        while next_batch is not None and item_processed < steps_per_epoch:
            if cuda_profile is not None and cuda_profile[0] == item_processed:
                utils.cudaProfilerStart()
            data, label = next_batch[0][0], next_batch[0][1]
            next_batch = _step(item_processed, data, label, 
                               performance_counter, iterator)
            item_processed += 1

            if cuda_profile is not None and cuda_profile[1] == item_processed:
                utils.cudaProfilerStop()
    else:
        for idx, batch_item in enumerate(dataset_iterator):
            if cuda_profile is not None and cuda_profile[0] == idx:
                utils.cudaProfilerStart()
            data, label = batch_item[0][0], batch_item[0][1]
            _step(idx, data, label, performance_counter)

            if cuda_profile is not None and cuda_profile[1] == idx:
                utils.cudaProfilerStop()

    dataset_iterator.reset()

    utils.logger.event(key='throughput',
                       value=performance_counter.throughput,
                       metadata={'epoch_num': epoch_number + 1})


def validate(model: mx.gluon.HybridBlock, dataset_iterator, batch_size: int, 
             epoch_number: int, use_fp16: bool = False, *,
             dist_desc: utils.DistributedEnvDesc, profile: bool = False) -> float:
    utils.logger.start(key=utils.logger.constants.EVAL_START, 
                       metadata={'epoch_num': epoch_number + 1})
    mae_metric = utils.DistributedMAE(dist_desc, sync=True)

    with utils.ProfilerSection("validate", profile=profile):
        for idx, batch_item in enumerate(dataset_iterator):
            #print(mx.autograd.is_training())
            data, label = batch_item[0][0], batch_item[0][1]
            if use_fp16:
                data = mx.nd.cast(data, "float16")
            
            output = model.model(data)

            if use_fp16:
                output = mx.nd.cast(output, "float32")
            mae_metric.update(label, output)
        dataset_iterator.reset()

        validation_metric = mae_metric.return_global()

    utils.logger.stop(key=utils.logger.constants.EVAL_STOP, 
                    metadata={'epoch_num': epoch_number + 1})
    return validation_metric


if __name__ == "__main__":
    parser = utils.ArgumentParser.build(argparse.ArgumentParser(
        description="NVIDIA MLPerf-HPC Cosmoflow benchmark implementation"))

    main(parser.parse_args())