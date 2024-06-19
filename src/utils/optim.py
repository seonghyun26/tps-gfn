import math

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import \
ExponentialLR, \
MultiStepLR, \
CosineAnnealingLR, \
ReduceLROnPlateau, \
CosineAnnealingWarmRestarts, \
_LRScheduler

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def set_optimizer(name, parameters, lr, args=None):
    name = name.lower()
    
    if name == None or name == "":
        print('No optimizer specified, using Adam')
        optimizer = Adam(parameters, lr=lr)
    elif name == 'sgd':
        optimizer = SGD(parameters, lr=lr)
    elif name == 'adam':
        optimizer = Adam(parameters, lr=lr)
    else:
        raise ValueError('Invalid optimizer')   
    
    return optimizer

def set_scheduler(name, optimizer, lr=0.0001, args=None):
    name = name.lower() if name is not None else None
    
    if name == None or name == "":
        print('No scheduler specified, using None')
        scheduler = None
    elif name == "exp":
        scheduler = ExponentialLR(optimizer, gamma=0.999)
    elif name == "multistep":
        scheduler = MultiStepLR(optimizer, milestones=[1000, 2000, 3000, 4000], gamma=0.75)
    elif name == "cosineanneal":
        scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=0.00001)
    elif name == "cosineannealwarmrestarts":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.num_rollouts, T_mult=2, eta_min=0.000001)
    elif name == "cosineannealwarmuprestarts":
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=args.num_rollouts, T_mult=1, T_up=50, eta_max=lr)
    elif name == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, 'min')
    else:
        raise ValueError(f'Scheduler: "{name}", this is an invalid Scheduler or to be implemented')
    
    return scheduler
        
    