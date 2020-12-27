import numpy as np

import torch

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
'''
from estimator import SizeEstimator

se = SizeEstimator(e, input_size=(181686,))
print(se.estimate_size())

# Returns
# (size in megabytes, size in bits)
# (408.2833251953125, 3424928768)

print(se.param_bits) # bits taken up by parameters
print(se.forward_backward_bits) # bits stored for forward and backward
print(se.input_bits) # bits for input
'''

train_skip_gram(e, train_x, exclude_tokens=(0,), verbose=True)
