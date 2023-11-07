from torch.optim.lr_scheduler import _LRScheduler
from collections import Counter

class MultiStepLRWarmup(_LRScheduler):

    def __init__(self, optimizer, warmup_steps, warmup_factor, milestones, gamma=0.1, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        self.milestones = Counter([x + self.warmup_steps + 1 for x in milestones])
        self.gamma = gamma
        super(MultiStepLRWarmup, self).__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        # linear warmup phase
        if self.last_epoch <= self.warmup_steps:
            if self.warmup_factor == 1.0:
                return [base_lr * (float(self.last_epoch) / self.warmup_steps)
                        for base_lr in self.base_lrs]
            else:
                return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.warmup_steps + 1.)
                        for base_lr in self.base_lrs]
                
        # decay phase
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
                for group in self.optimizer.param_groups]

