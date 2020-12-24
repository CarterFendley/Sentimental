import os
import json
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

from imdb_data import is_valid_cache, save_processed_data, load_processed_data

DIRNAME = os.path.dirname(__file__)
DEFAULT_SAVE_DIR = os.path.join(DIRNAME, 'data/saved_models')

############################
# Saving / Loading Helpers #
############################

def gen_postfix(path):
    post_fix = 0
    while True:
        if post_fix == 0 and os.path.exists(path):
            post_fix +=1
        elif os.path.exists(f'{path}-{post_fix}'):
            post_fix +=1
        else:
            break
    if post_fix != 0:
        path += f'-{post_fix}'

    return path

def is_valid_model_save(path):
    # Check model files exist 
    model_files = (
        f'{path}/model_trained.pt',
        f'{path}/model_config.json'
    )

    for f in model_files:
        if not os.path.isfile(f):
            return False

    # Return value of data cache validation 
    return is_valid_cache(f'{path}/data_cache')

def make_model(config, data):
    return SentimentLSTM(
        len(data['vocab_to_int']),
        config['output_size'],
        config['embedding_dim'],
        config['hiddem_dim'],
        config['n_layers']
    )

def save_model(model, config, data, save_dir=DEFAULT_SAVE_DIR, _cache_dir=None, name=None, time_stamp=True):
    if name is None:
        if time_stamp:
            name = f'SentimentLSTM {datetime.now()}'
        else:
            name = 'SentimentLSTM'

    save_path = os.path.join(save_dir, name)
    save_path = gen_postfix(save_path)

    # Create model directory
    os.makedirs(save_path)

    # Save model & config
    torch.save(model.state_dict(), f'{save_path}/model_trained.pt')
    with open(f'{save_path}/model_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Save link to cached data
    save_processed_data(data, f'{save_path}/data_cache', _cache_dir=_cache_dir)

def load_model(save_path):
    assert is_valid_model_save(save_path)

    # Load config
    config = None
    with open(f'{save_path}/model_config.json', 'r') as f:
        config = json.load(f)

    # Load data
    data = load_processed_data(cache_path=f'{save_path}/data_cache')

    vocab_size = len(data['vocab_to_int'])

    model = make_model(config, data)
    model.load_state_dict(torch.load(f'{save_path}/model_trained.pt'))

    return model, config, data

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
    
    def forward(self, x):
        batch_size = x.size(0)

        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)

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
        return sig_out
    
    def init_hidden(self, batch_size, device):
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

def score_eval_batch(model, batch, criterion, device):
    losses = []
    correct_count, total_count = 0, 0

    model.eval()
    with torch.no_grad():
        reviews, labels = batch

        # Put inputs in right type and on right device
        reviews = reviews.type(torch.LongTensor) # For embedding layer
        reviews, labels = reviews.to(device), labels.to(device)

        # Send inputs through model
        output = model(reviews)

        # Calc loss
        loss = criterion(output, labels.float())
        losses.append(loss.item())

        # Round to get predictions
        pred = torch.round(output)
        correct = pred.eq(labels.view_as(pred))
        
        correct_count += np.sum(correct.to('cpu').numpy())
        total_count += labels.size(0)
    
    accuracy = round(correct_count / total_count, 5)

    model.train()

    return {
        'losses': losses,
        'correct': correct_count,
        'total': total_count,
        'accuracy': accuracy
    }

def score_eval_set(model, loader, criterion, device):
    losses = []
    correct = 0
    total = 0
    
    for reviews, labels in loader:
        batch_score = score_eval_batch(model, (reviews, labels), criterion, device)

        losses.extend(batch_score['losses'])
        correct += batch_score['correct']
        total += batch_score['total']
    
    return {
        'losses': losses,
        'correct': correct,
        'total': total,
        'accuracy': round(correct / total, 5),
        'mean_loss': round(np.mean(losses), 5)
    }