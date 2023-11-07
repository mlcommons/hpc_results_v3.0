import tensorflow as tf
from mpi4py import MPI
import numpy as np
from mlperf_logging import mllog
import os

class CustomInMemoryEvaluatorHook(tf.estimator.experimental.InMemoryEvaluatorHook):
    def __init__(self,
                 estimator,
                 input_fn,
                 steps=None,
                 hooks=None,
                 name=None,
                 every_n_iter=100,
                 initial_epoch=0,
                 target_mae=0):
        super().__init__(estimator,
                         input_fn,
                         steps=steps,
                         hooks=hooks,
                         name=name,
                         every_n_iter=every_n_iter)
        self._epoch = initial_epoch
        self._mllogger = mllog.get_mllogger()
        self._world_rank = MPI.COMM_WORLD.Get_rank()
        self._world_size = MPI.COMM_WORLD.Get_size()
        self._target_mae = target_mae

    def begin(self):
        super().begin()
        self._eval_mae = None

    def _evaluate(self, train_session):
        if self._iter_count > 0:
            self._custom_evaluate(train_session)
        else:
            self._timer.update_last_triggered_step(self._iter_count)

    def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
        """Runs evaluator."""
        self._iter_count += 1
        if self._timer.should_trigger_for_step(self._iter_count):
            self._evaluate(run_context.session)
            if self._eval_mae and self._eval_mae <= self._target_mae:
                run_context.request_stop()

    def end(self, session):
        pass

    def _custom_evaluate(self, train_session):
        self._mllog_begin()

        var_name_to_value = train_session.run(self._var_name_to_train_var)
        placeholder_to_value = {
            self._var_name_to_placeholder[v_name]: var_name_to_value[v_name]
            for v_name in var_name_to_value
        }

        def feed_variables(scaffold, session):
            del scaffold
            session.run(self._var_feed_op, feed_dict=placeholder_to_value)

        scaffold = tf.compat.v1.train.Scaffold(
            init_fn=feed_variables, copy_from_scaffold=self._scaffold)

        with self._graph.as_default():
            eval_result = self._estimator._evaluate_run(
                checkpoint_path=None,
                scaffold=scaffold,
                update_op=self._update_op,
                eval_dict=self._eval_dict,
                all_hooks=self._all_hooks,
                output_dir=self._eval_dir)

        metrics = np.array([eval_result['loss'], eval_result['mae']], np.float32)
        acc_metrics = self._allreduce_mean(metrics)
        eval_result['loss'] = acc_metrics[0]
        eval_result['mae'] = acc_metrics[1]

        self._eval_mae = eval_result['mae']
        self._mllog_end(eval_result)
        self._timer.update_last_triggered_step(self._iter_count)
        self._epoch += 1

    def _mllog_begin(self):
        if self._world_rank == 0:
            self._mllogger.start(key=mllog.constants.EVAL_START,
                                metadata={'epoch_num': self._epoch + 1})

    def _mllog_end(self, eval_result):
        global_step = eval_result['global_step']

        if self._world_rank == 0:
            self._mllogger.end(key=mllog.constants.EVAL_STOP,
                               metadata={'epoch_num': self._epoch + 1})
            self._mllogger.event(key='eval_error', value=eval_result['mae'],
                                 metadata={'epoch_num': self._epoch + 1})

    def _allreduce_mean(self, metrics):
        acc_metrics = np.empty_like(metrics)
        MPI.COMM_WORLD.Allreduce(metrics, acc_metrics, MPI.SUM)
        return acc_metrics / self._world_size

class EpochLogHook(tf.compat.v1.train.SessionRunHook):
    def __init__(self,
                 train_epoch_steps,
                 initial_epoch=0):
        self._initial_epoch = initial_epoch
        self._mllogger = mllog.get_mllogger()
        self._train_epoch_steps = train_epoch_steps
        self._world_rank = MPI.COMM_WORLD.Get_rank()

    def begin(self):
        self._iter_count = 0
        self._epoch = self._initial_epoch

    def before_run(self, session):
        if self._world_rank == 0 and self._iter_count % self._train_epoch_steps == 0:
            self._mllogger.start(key=mllog.constants.EPOCH_START,
                                 metadata={'epoch_num': self._epoch + 1})

    def after_run(self, run_context, run_values):
        if self._world_rank == 0 and (self._iter_count + 1) % self._train_epoch_steps == 0:
            self._mllogger.end(key=mllog.constants.EPOCH_STOP,
                               metadata={'epoch_num': self._epoch + 1})
            self._epoch += 1
        self._iter_count += 1

        #if self._iter_count %100 == 0:
        #    cmd = "./get_thr_aff.sh {}".format(os.getpid())
        #    os.system(cmd)

