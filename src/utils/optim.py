from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import \
ExponentialLR, \
MultiStepLR, \
CosineAnnealingLR, \
ReduceLROnPlateau, \
CosineAnnealingWarmRestarts


def set_optimizer(name, parameters, lr, args=None):
    name = name.lower()
    
    if name == None or "":
        print('No optimizer specified, using Adam')
        optimizer = Adam(parameters, lr=lr)
    elif name == 'sgd':
        optimizer = SGD(parameters, lr=lr)
    elif name == 'adam':
        optimizer = Adam(parameters, lr=lr)
    else:
        raise ValueError('Invalid optimizer')   
    
    return optimizer

def set_scheduler(name, optimizer, args=None):
    name = name.lower()
    
    if name == None or "":
        print('No scheduler specified, using None')
        scheduler = None
    if name == "exp":
        scheduler = ExponentialLR(optimizer, gamma=0.96)
    elif name == "multistep":
        scheduler = MultiStepLR(optimizer, milestones=[1000, 2000, 3000, 4000], gamma=0.75)
    elif name == "cosineanneal":
        scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=0.00001)
    elif name == "cosineannealwarmrestarts":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=400, T_mult=2, eta_min=0.00001)
    elif name == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, 'min')
    else:
        raise ValueError('Invalid Scheduler or to be implemented')
    
    return scheduler
        
    