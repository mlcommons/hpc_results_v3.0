import mxnet as mx
import mxnet.gluon.nn as nn
import horovod.mxnet as hvd

from mxnet.contrib import amp
from mxnet.gluon.contrib.nn import SpatialParallelConv3D, SpatialParallelAllgather, SpatialParallelSplit

import numpy as np

import utils

from typing import Literal, Optional


class DefaultInitializer(mx.init.Xavier):
    def __init__(self, layout=Literal["NCDHW", "NDHWC"]):
        super().__init__()
        self.layout=layout

    def _init_weight(self, name, arr):
        if len(arr.shape) != 5:
            return super()._init_weight(name, arr)
        self._init_conv3d_weight(arr)
        

    def _init_conv3d_weight(self, arr):
        if self.layout == "NDHWC":
            fan_in, fan_out = arr.shape[-1], arr.shape[0]
            prod = np.prod(arr.shape[1:-1])
        else:
            fan_in, fan_out = arr.shape[1], arr.shape[0]
            prod = np.prod(arr.shape[2:])
        
        scale = np.sqrt(6.0 / ((fan_in + fan_out) * prod))
        mx.nd.random.uniform(-scale, scale, shape=arr.shape, out=arr)
        


class PaddedConvolution(nn.HybridBlock):
    def __init__(self, conv_channels: int, kernel_size: int,
                 layout: Optional[Literal["NCDHW", "NDHWC"]] = None,
                 spatial: Optional[int] = None):
        super().__init__()
        if layout is None:
            layout = "NCDHW" if kernel_size & 1 == 0 else "NDHWC"
        in_layer_padding = (kernel_size // 2) if (kernel_size & 1) else 0

        self.channels = conv_channels

        conv3d_arguments = {
            "channels": conv_channels,
            "kernel_size": kernel_size,
            "padding": in_layer_padding,
            "layout": layout
        }
        
        with self.name_scope():
            
            self.kernel_size = kernel_size
            if spatial is None or spatial == 1:
                self.convolution = nn.Conv3D(**conv3d_arguments)
            else:
                self.convolution = SpatialParallelConv3D(**conv3d_arguments,
                                                         num_gpus=spatial)
            self.activation = nn.LeakyReLU(alpha=0.3)

    def hybrid_forward(self, F, x):
        if self.kernel_size & 1 == 0:
            x = F.pad(x, mode="constant", constant_value=0,
                      pad_width=(0, 0, 0, 0, 0, self.kernel_size // 2,
                                 0, self.kernel_size // 2,
                                 0, self.kernel_size // 2))
        result = self.activation(self.convolution(x))
        return result


class Scale1p2(nn.HybridBlock):
    def __init__(self):
        super().__init__()

    def hybrid_forward(self, F, x):
        return x * 1.2


def add_dense_block(network_container: nn.HybridSequential, 
                    dense_unit: int, lrelu_alpha: float, 
                    dropout_rate: Optional[float] = None):
    network_container.add(nn.Dense(dense_unit),
                          nn.LeakyReLU(alpha=lrelu_alpha))
    if dropout_rate is not None:
        network_container.add(nn.Dropout(dropout_rate))


def build_cosmoflow_model(dist_desc: utils.DistributedEnvDesc,
                          n_conv_layers: int = 5,
                          conv_kernel: int = 2,
                          dropout_rate: float = 0.5,
                          layout: Literal["NCDHW", "NDHWC"] = "NDHWC",
                          use_wd: bool = False,
                          spatial_span: int = 1) -> nn.HybridSequential:
    cosmoflow = nn.HybridSequential()

    if spatial_span > 1:
        cosmoflow.add(SpatialParallelSplit(num_gpus=spatial_span))

    for i in range(n_conv_layers):
        cosmoflow.add(PaddedConvolution(conv_channels=32 * (1 << i),
                                        kernel_size=conv_kernel,
                                        layout=layout, 
                                        spatial=spatial_span),
                      nn.MaxPool3D(pool_size=2, layout=layout))

    if spatial_span > 1:
        cosmoflow.add(SpatialParallelAllgather(num_gpus=spatial_span))

    dropout_rate = dropout_rate if not use_wd else None
    add_dense_block(cosmoflow, 128, 0.3, dropout_rate)
    add_dense_block(cosmoflow, 64, 0.3, dropout_rate)

    cosmoflow.add(nn.Dense(4),
                  nn.Activation("tanh"),
                  Scale1p2())

    return cosmoflow
                                    
class CosmoflowWithLoss(nn.HybridBlock):
    def __init__(self, 
                 dist_desc: utils.DistributedEnvDesc,
                 n_conv_layers: int = 5,
                 conv_kernel: int = 2,
                 dropout_rate: float = 0.5,
                 layout: Literal["NCDHW", "NDHWC"] = "NDHWC",
                 use_wd: bool = False,
                 spatial_span: int = 1):
        super().__init__()
        self.layout = layout
        self.spatial = spatial_span
        self.model = build_cosmoflow_model(dist_desc, n_conv_layers, conv_kernel,
                                           dropout_rate, layout, use_wd, spatial_span)
        self.loss_fn = mx.gluon.loss.L2Loss()

    def hybrid_forward(self, F, x, y_true):
        y_pred = self.model(x)#.astype(np.float32)
        return self.loss_fn(y_pred, y_true)

    def init(self, ctx, batch_size: int, 
             use_amp: bool, use_wd: bool, 
             dist_desc: utils.DistributedEnvDesc,
             checkpoint: Optional[str] = None,
             dtype: str = "float32"):

        self.model.initialize(init=DefaultInitializer(layout=self.layout),
                              ctx=ctx)

        input_shape = (batch_size, 4, 128, 128, 128) if self.layout == "NCDHW" \
                else (batch_size, 128, 128, 128, 4)
        random_batch = (mx.nd.random.uniform(shape=input_shape, 
                                            ctx=ctx,
                                            dtype=dtype),
                        mx.nd.random.uniform(shape=(batch_size, 4),
                                            ctx=ctx,
                                            dtype="float32"))
        
        #if dist_desc.master:
        #    self.model.summary(random_batch[0])
            

        #self.hybridize(static_alloc=True, static_shape=True)
        self.model.hybridize(static_alloc=True, static_shape=True)
        self.warmup(ctx, dist_desc, use_amp, batch_size)

        if checkpoint is not None and checkpoint != "":
            self.load_parameters(checkpoint, ctx=ctx)
        else:
            self.model.initialize(init=DefaultInitializer(layout=self.layout),
                                  ctx=ctx, force_reinit=True)
        if use_wd:
            self.model.collect_params(select=".*conv.*_weight|.*_bias").setattr("wd_mult", 0.0)

        if self.spatial > 1:
            self.model.collect_params(select=".*dense.*").setattr('lr_mult', 1.0 / self.spatial)
        self.model.hybridize(static_alloc=True, static_shape=True)

    def warmup(self, ctx, dist_desc: utils.DistributedEnvDesc, use_amp: bool, batch_size: int):
        fake_optimizer = mx.optimizer.SGD(learning_rate=1e-5,
                                          momentum=0.9)
        if dist_desc.size == 1:
            fake_trainer = mx.gluon.Trainer(self.collect_params(), fake_optimizer,
                                            update_on_kvstore=False)
        else:
            fake_trainer = hvd.DistributedTrainer(self.collect_params(), fake_optimizer,
                                                  num_groups=1)

        for _ in range(100):
            input_shape = (batch_size, 4, 128, 128, 128) if self.layout == "NCDHW" \
                else (batch_size, 128, 128, 128, 4)
            data = mx.nd.random.uniform(shape=input_shape, 
                                        ctx=ctx,
                                        dtype="float32")
            label = mx.nd.random.uniform(shape=(batch_size, 4),
                                         ctx=ctx,
                                         dtype="float32")
            with mx.autograd.record():
                output = self(data, label)
                mx.autograd.backward(output)
            fake_trainer.step(batch_size)
            _ = output.asnumpy()
        mx.nd.waitall()

