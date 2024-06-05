from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau

def set_optimizer(name, parameters, lr):
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

def set_scheduler(name, optimizer):
    name = name.lower()
    
    if name == None or "":
        print('No scheduler specified, using None')
        scheduler = None
    if name == "exp":
        scheduler = ExponentialLR(optimizer, gamma=0.9)
    elif name == "multistep":
        scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
    elif name == "cosineanneal":
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
    elif name == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, 'min')
    else:
        raise ValueError('Invalid Scheduler or to be implemented')
    
    return scheduler
        
    