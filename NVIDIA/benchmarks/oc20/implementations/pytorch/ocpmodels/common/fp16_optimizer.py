import itertools

import torch
from apex.multi_tensor_apply import multi_tensor_applier


class FP16_Optimizer(object):
    def __init__(self, optimizer):
        self.optimizer = optimizer

        # Create master FP32 weights for FP16 model weights.
        self.fp16_groups, self.fp32_from_fp16_groups = [], []
        for i, param_group in enumerate(self.optimizer.param_groups):
            fp16_params_this_group, fp32_from_fp16_params_this_group = [], []
            for i, param in enumerate(param_group["params"]):
                if param.requires_grad:
                    if param.dtype == torch.half:
                        fp16_params_this_group.append(param)
                        master_param = param.detach().float()
                        master_param.requires_grad = True
                        param_group["params"][i] = master_param
                        fp32_from_fp16_params_this_group.append(master_param)
                    elif param.dtype == torch.float:
                        param_group["params"][i] = param
                    else:
                        raise TypeError(f"Received {param.type()} but expected fload or half")

            self.fp16_groups.append(fp16_params_this_group)
            self.fp32_from_fp16_groups.append(fp32_from_fp16_params_this_group)

        self.all_fp16_params = list(itertools.chain(*self.fp16_groups))
        self.all_fp32_from_fp16_params = list(itertools.chain(*self.fp32_from_fp16_groups))

        if multi_tensor_applier.available:
            import amp_C

            self.multi_tensor_scale = amp_C.multi_tensor_scale
            self._dummy_overflow_buf = torch.cuda.IntTensor([0])

    def zero_grad(self, set_grads_to_None=False):
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                p.grad = None

        # Zero fp16 gradients owned by the model:
        for fp16_group in self.fp16_groups:
            for p in fp16_group:
                p.grad = None

    def _master_params_to_model_params(self):
        if len(self.all_fp16_params) == 0:
            return
        multi_tensor_applier(
            self.multi_tensor_scale,
            self._dummy_overflow_buf,
            [self.all_fp32_from_fp16_params, self.all_fp16_params],
            1.0,
        )

    @torch.no_grad()
    def step(self):
        self.update_master_grads()
        self.optimizer.step()
        self._master_params_to_model_params()

    def update_master_grads(self):
        if len(self.all_fp16_params) > 0:
            for model_param, master_param in zip(self.all_fp16_params, self.all_fp32_from_fp16_params):
                if model_param.grad is not None:
                    master_param.grad = model_param.grad.detach().float()
