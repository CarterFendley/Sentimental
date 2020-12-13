import numpy as np

import torch
from torchsummary import summary

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from skip_gram import SkipGram, train_skip_gram

print('\n-\tPre-training the embedding layer\n')

# Load train_y
train_x = np.load('/home/carter/src/TDS-LSTM-Tutorial/train_x.npy')

e = SkipGram(181686)
print(e)
summary(e, (1, 181686), device='cpu', dtypes=(torch.long,))
train_skip_gram(e, train_x, exclude_tokens=(0,), verbose=True, batch_size=1)
