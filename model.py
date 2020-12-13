#!/usr/bin/env python3

import os
print('\n-\tLoading py torch')
import torch
from string import punctuation

dev = 'cpu'

if torch.cuda.is_available():
    dev = 'cuda'

device = torch.device(dev)

POSITIVE = None
NEGATIVE = None

def format_review(review):
    review = review.lower()
    review = ''.join([c for c in review if c not in punctuation])

    return review

def load_set(directory):
    out = []
    for filepath in os.listdir(directory):
        #print(filepath)
        with open(f'{directory}/{filepath}', 'r') as f:
            out.append(format_review(f.read()))
    return out

##############
# Load Data
##############
print('\n-\tLoading Data')

POSITIVE = load_set('data/train/pos')
POSITIVE.extend(load_set('data/test/pos'))
NEGATIVE = load_set('data/train/neg')
NEGATIVE.extend(load_set('data/test/neg'))

COMBINED = POSITIVE[:]
COMBINED.extend(NEGATIVE)

print(f'Number of reviews :', len(COMBINED))

############
# Tokenize and encode words
# ############

print('\n-\tTokenizing Words')

# Creat sorted lis of words
from collections import Counter

all_text = ' '.join(COMBINED)
words = all_text.split()

count_words = Counter(words)
total_words = len(words)
sorted_words = count_words.most_common(total_words) # total_words specificies how many most common to return (all of them)

print(f"Top five most common words: ")
print(f"\t{sorted_words[0:4]}")

# Create a word -> int mapping dictionary
vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)} # Plus one to reserve 0 value for padding


def tokenize_set(s, mapping):
    set_int = []
    for x in s:
        x_int = [mapping[w] for w in x.split()]

        set_int.append(x_int)
    
    return set_int

POSITIVE = tokenize_set(POSITIVE, vocab_to_int)
NEGATIVE = tokenize_set(NEGATIVE, vocab_to_int)

print("Example tokenized review")
print(f'\t{POSITIVE[32]}')

# TODO: Maybe combine stuff?

POSTIVE_len = [len(x) for x in POSITIVE]
NEGATIVE_len = [len(x) for x in NEGATIVE]

if False:
    print('\n-\tPreforming padanas analysis')

    import pandas as pd
    import matplotlib.pyplot as plt
    #%matplotlib inline

    reviews_len = POSTIVE_len[:]
    reviews_len.extend(NEGATIVE_len)
    pd.Series(reviews_len).hist()
    plt.show()

    print(pd.Series(reviews_len).describe())

    

###############
# Messaging
###############

import numpy as np

def remove_reviews(reviews_int, reviews_len):
    reviews_int = [reviews_int[i] for i, l in enumerate(reviews_len) if l > 0]
    reviews_len = [reviews_len[i] for i, l in enumerate(reviews_len) if l > 0]
    return (reviews_int, reviews_len)

def pad_truncate(reviews_int, reviews_len, seq_length):
    features = np.zeros((len(reviews_int), seq_length), dtype=int)

    for i, review in enumerate(reviews_int):
        review_len = len(review)

        if review_len <= seq_length:
            zeroes = list(np.zeros(seq_length - review_len))
            new = zeroes+review
        elif review_len > seq_length:
            new = review[0:seq_length]
        
        features[i,:] = np.array(new)
    return features



print('\n-\tFormating data\n')

print("Pre-filtering:")
print(f"\t Positive reviews: {len(POSITIVE)}")
print(f'\t Negative reviews: {len(NEGATIVE)}')

POSITIVE, POSTIVE_len = remove_reviews(POSITIVE, POSTIVE_len)
NEGATIVE, NEGATIVE_len = remove_reviews(NEGATIVE, NEGATIVE_len)


POSITIVE = pad_truncate(POSITIVE, POSTIVE_len, 400)
NEGATIVE = pad_truncate(NEGATIVE, NEGATIVE_len, 400)

print("Post-filtering:")
print(f"\t Positive reviews: {len(POSITIVE)}")
print(f'\t Negative reviews: {len(NEGATIVE)}')

COMBINED = np.concatenate((POSITIVE, NEGATIVE), axis=0) # Concatonate the 2D array

COMBINED_LABELS = [1]*len(POSTIVE_len) 
COMBINED_LABELS.extend([0]*len(NEGATIVE_len))
COMBINED_LABELS = np.array(COMBINED_LABELS)
COMBINED_len = len(POSTIVE_len) + len(NEGATIVE_len)

print("Combined:")
print(f'\tReviews: {len(COMBINED)}')
print(f'\tLabels: {len(COMBINED_LABELS)}')

split_frac = 0.8
train_x = COMBINED[0:int(split_frac*COMBINED_len)]
train_y = COMBINED_LABELS[0:int(split_frac*COMBINED_len)]

remaining_x = COMBINED[int(split_frac*COMBINED_len):]
remaining_y = COMBINED_LABELS[int(split_frac*COMBINED_len):]

print("remaining", len(remaining_x))
print(len(remaining_y))

valid_x = remaining_x[0:int(len(remaining_x)*0.5)]
valid_y = remaining_y[0:int(len(remaining_y)*0.5)]

test_x = remaining_x[int(len(remaining_x)*0.5):]
test_y = remaining_y[int(len(remaining_y)*0.5):]

print("Train set:")
print(f"\tFeatures: {len(train_x)}")
print(f"\tLabels: {len(train_y)}")
print("Validate set:")
print(f"\tFeatures: {len(valid_x)}")
print(f"\tLabels: {len(valid_y)}")
print("Test set:")
print(f"\tFeatures: {len(test_x)}")
print(f"\tLabels: {len(test_y)}")


print('\n-\t Creating data loaders')
from torch.utils.data import DataLoader, TensorDataset

# Create tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

# Create data loaders
batch_size = 50
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

data_iter = iter(train_loader)
sample_x, sample_y = data_iter.next()

print('Sample input size: ', sample_x.size()) # batch_size, seq_length
print('Sample input: \n', sample_x)
print()
print('Sample label size: ', sample_y.size()) # batch_size
print('Sample label: \n', sample_y)

########################
# Model
########################

print('\n-\tBuilding model\n')

import torch.nn as nn

class SentimentLSTM(nn.Module):
    '''
    The RNN model
    '''

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super().__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()
    
    def forward(self, x, hidden):
        batch_size = x.size(0)

        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        '''
        View changes the shape of the tensor. From the docs,
        it appears that contigous is called because there are
        Senarios in which view will fail to reshaped. 

        TODO: Reshape better?

        Source: https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view
        ''' 
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # Dropout and fully connected
        out = self.dropout(lstm_out)
        out = self.fc(out)

        sig_out = self.sig(out)

        # Reshape to be batch size
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # Get last batch of labels

        # Return sigmoid output and hidden state
        return sig_out, hidden
    
    def init_hidden(self, batch_size):
        '''
        Initalize the hidden state

        Create two tensors of shape (n_layers * batch_size * hidden_dim) for:
            - hidden state
            - cell state
        
        '''

        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                    weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        
        return hidden

print('\n-\tInstantiating model\n')

vocab_size = len(vocab_to_int)+1 # +1 for 0 padding
output_size = 1
embedding_dim = 400
hidden_dim = 500
n_layers = 2

net = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

print('Network:')
print(net)

########################
# Training 
########################

if True:
    print('\n-\tPre-training the embedding layer\n')


    from skip_gram import SkipGram
    e = SkipGram(vocab_size)
    e.train(train_x, verbose=True)

print('\n-\tTraining the model\n')
# Loss and optimization functions
lr = 0.001 # Learning rate
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# Training params
epochs = 4 # TODO: Play with this and look validation loss
counter = 0
print_every = 100
clip = 5 # gradient clipping TODO:What?


net.to(device)

net.train()
for e in range(epochs):
    h = net.init_hidden(batch_size) # Init hidden state

    # Batch loops
    for inputs, labels in train_loader:
        counter += 1

        # Create new vars for the hidden state
        # Prevents backprop through entire training history TODO: What?
        h = tuple([each.data for each in h])

        # zero accumulate gradients TODO:WHAT?
        net.zero_grad()

        # Run the model
        inputs = inputs.type(torch.LongTensor) # Put inputs into long int form (needed for embedding layer lookup)

        inputs, labels = inputs.to(device), labels.to(device)
        output, h = net(inputs, h)

        # Calculate loss and backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # clip_grad_normal helps prevent the exploding gradient problem in RNNs / LSTM
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # Loss stats
        if counter % print_every == 0:
            print('Calculating validation...')
            # Get calidation loss
            val_h = net.init_hidden(batch_size)
            val_losses = []
            
            net.eval() # TODO: What?

            for inputs, labels in valid_loader:
                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])
                
                inputs = inputs.type(torch.LongTensor)
                inputs, labels = inputs.to(device), labels.to(device)
                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())
            
            net.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))

torch.save(net, '/home/carter/src/TDS-LSTM-Tutorial/model.pt')

########################
# Testing 
########################

print('\n-\Testing the model\n')

test_losses = [] # Track loss
num_correct = 0

# init hidden state
h = net.init_hidden(batch_size)

net.eval()
# Interate over test data
for inputs, labels in test_loader:
    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    h = tuple([each.data for each in h])

    
    # get predicted outputs
    inputs = inputs.type(torch.LongTensor)
    inputs, labels = inputs.to(device), labels.to(device)
    output, h = net(inputs, h)
    
    # calculate loss
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    
    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())  # rounds to the nearest integer
    
    # compare predictions to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.to('cpu').numpy())
    num_correct += np.sum(correct)


# -- stats! -- ##
# avg test loss
print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))
