import torch

def get_device(prefer_gpu=True):
    dev = 'cpu'

    if prefer_gpu and torch.cuda.is_available():
        dev = 'cuda'
    
    return torch.device(dev)