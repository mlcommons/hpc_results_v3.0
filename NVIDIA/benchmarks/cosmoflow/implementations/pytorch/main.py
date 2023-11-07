# Copyright (c) 2021-2023 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
from typing import Any
import utils

from data.dali_npy import NPyLegacyDataPipeline
from data.dali_tfr_gzip import TFRecordDataPipeline
from data.dali_synthetic import SyntheticDataPipeline
from utils.app import PytorchApplication
from omegaconf import OmegaConf

import hydra
import torch
import torch.distributed

from torch.nn.parallel import DistributedDataParallel as DDP
from utils.executor import get_executor_from_config

from model.cosmoflow import get_standard_cosmoflow_model, Convolution3DLayout

from trainer import Trainer
from optimizer import get_optimizer


class CosmoflowMain(PytorchApplication):
    def setup(self) -> None:
        super().setup()

        with utils.ProfilerSection("initialization", profile=self._config["profile"]):
            super().init_ddp()

            utils.logger.event(key=utils.logger.constants.CACHE_CLEAR)
            utils.logger.start(key=utils.logger.constants.INIT_START)

            num_nodes = int(os.getenv("DGXNNODES", self._distenv.size // self._distenv.local_size))
            utils.logger.mllogger.mlperf_submission_log(
                benchmark="cosmoflow", num_nodes=num_nodes)

            utils.logger.event(key="number_of_nodes",
                               value=self._distenv.size // self._distenv.local_size)
            utils.logger.event(key="accelerators_per_node",
                               value=self._distenv.local_size)

            model_cfg = self._config["model"]
            train_cfg = model_cfg["training"]

            if "seed" in self._config:
                seed = self._config["seed"] + self._distenv.instance
            else:
                seed = random.randint(0, 65536)
                if not self._distenv.is_single:
                    seed = self._distenv.master_mpi_comm.bcast(
                        seed, root=0) + self._distenv.instance
            utils.logger.event(key=utils.logger.constants.SEED, value=seed)

            assert (train_cfg["weight_decay"] == 0.0 or train_cfg["dropout_rate"] ==
                    0.0), "Both 'weight_decay' and 'dropout_rate' cannot be different from 0"

            if self._config["data"]["dataset"] == "synthetic":
                self._training_pipeline = SyntheticDataPipeline(config=self._config["data"],
                                                                distenv=self._distenv,
                                                                sample_count=self._config["data"]["train_samples"],
                                                                device=self._distenv.local_rank)
                self._validation_pipeline = SyntheticDataPipeline(config=self._config["data"],
                                                                  distenv=self._distenv,
                                                                  sample_count=self._config["data"]["valid_samples"],
                                                                  device=self._distenv.local_rank)
            elif self._config["data"]["dataset"] == "cosmoflow_npy":
                self._training_pipeline, self._validation_pipeline = NPyLegacyDataPipeline.build(config=self._config["data"],
                                                                                                 distenv=self._distenv,
                                                                                                 device=self._distenv.local_rank,
                                                                                                 seed=seed)
            elif self._config["data"]["dataset"] == "cosmoflow_tfr":
                self._training_pipeline, self._validation_pipeline = TFRecordDataPipeline.build(config=self._config["data"],
                                                                                                distenv=self._distenv,
                                                                                                device=self._distenv.local_rank,
                                                                                                seed=seed)

            model_layout = Convolution3DLayout(model_cfg["layout"])
            self._model = get_standard_cosmoflow_model(kernel_size=model_cfg["conv_layer_kernel"],
                                                       n_conv_layer=model_cfg["conv_layer_count"],
                                                       n_conv_filters=model_cfg["conv_layer_filters"],
                                                       dropout_rate=train_cfg["dropout_rate"],
                                                       layout=model_layout,
                                                       script=model_cfg["script"],
                                                       device="cuda")
            utils.logger.event(key="dropout", value=train_cfg["dropout_rate"])

            capture_stream = torch.cuda.Stream()

            if not self._distenv.is_single:
                with torch.cuda.stream(capture_stream):
                    self._model = DDP(self._model,
                                      device_ids=[self._distenv.local_rank],
                                      process_group=None)

            self._optimizer, self._lr_scheduler = get_optimizer(
                train_cfg, self._model)

            utils.logger.event(key=utils.logger.constants.GLOBAL_BATCH_SIZE,
                               value=self._config["data"]["batch_size"] * self._distenv.size)
            utils.logger.event(key=utils.logger.constants.TRAIN_SAMPLES,
                               value=len(self._training_pipeline))
            utils.logger.event(key=utils.logger.constants.EVAL_SAMPLES,
                               value=len(self._validation_pipeline))
            self._trainer = Trainer(self._config,
                                    self._model,
                                    self._optimizer,
                                    self._lr_scheduler,
                                    distenv=self._distenv,
                                    amp=train_cfg["amp"],
                                    enable_profiling=self._config["profile"])
            self._trainer.warmup(capture_stream=capture_stream)

            self._stager_executor = get_executor_from_config(
                self._distenv, self._config)

            self._run_stop_printed = False

    def stop_training(self, status: str, epoch_num: int, time: int):
        utils.logger.mllogger.log_run_stop(status=status,
                                           time=time,
                                           epoch_num=epoch_num)
        with torch.no_grad():
            for group in self._optimizer.param_groups:
                if isinstance(group["lr"], torch.Tensor):
                    group["lr"].fill_(0.0)
                else:
                    group["lr"] = 0.0

        for lmbd in self._lr_scheduler.lr_lambdas:
            lmbd.disable()

        self._run_stop_printed = True

    def run(self) -> Any:
        model_cfg = self._config["model"]
        eval_only = "eval_only" in self._config

        utils.logger.start(key="staging_start")
        with utils.ExecutionTimer(name="data_staging",
                                  profile=self._config["profile"]) as staging_timer:
            with self._stager_executor:
                wait_train = self._training_pipeline.stage_data(self._stager_executor,
                                                                profile=self._config["profile"])
                wait_eval = self._validation_pipeline.stage_data(self._stager_executor,
                                                                profile=self._config["profile"])

                if wait_train is not None:
                    wait_train()
                if wait_eval is not None:
                    wait_eval()
            self._distenv.local_barrier()
        utils.logger.stop(key="staging_stop",
                          metadata={"staging_duration": staging_timer.time_elapsed()})


        self._distenv.global_barrier()
        utils.logger.mllogger.log_init_stop_run_start()

        run_status = None
        with utils.ExecutionTimer(name="run_time", profile=self._config["profile"]) as run_time:
            train_iterator = iter(self._training_pipeline)
            val_iterator = iter(self._validation_pipeline)

            for epoch in range(model_cfg["training"]["train_epochs"] * 10):
                last_score = self._trainer.epoch_step(
                    train_iterator, val_iterator, epoch, eval_only=eval_only)

                if last_score <= model_cfg["training"]["target_score"]:
                    run_status = "success"
                    
                    if ("early_stop" in model_cfg["training"] and
                        model_cfg["training"]["early_stop"]):
                        break
                    elif not self._run_stop_printed:
                        self.stop_training(run_status, epoch+1, run_time.time_elapsed())

                # Run for at least 13 minutes and train_epochs are reach
                if run_time.time_elapsed() > 11 * 60 and (
                    run_status is not None or epoch > model_cfg["training"]["train_epochs"]):
                    break
            
            if run_status is None:
                run_status = "aborted"

            torch.cuda.synchronize()
        self._distenv.local_barrier()

        if not self._run_stop_printed:
            utils.logger.mllogger.log_run_stop(status=run_status,
                                               time=run_time.time_elapsed(),
                                               epoch_num=epoch+1)
        self._distenv.global_barrier()


@hydra.main(config_path="configs",
            config_name="baseline",
            version_base=None)
def main(cfg: OmegaConf) -> Any:
    return CosmoflowMain(cfg).exec()


if __name__ == "__main__":
    main()
