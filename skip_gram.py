import os
import numpy as np
import json
import torch
from torch import nn, optim
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MultiLabelBinarizer

from tqdm import tqdm

from util.file_helpers import gen_postfix

DEFAULT_CONTEXT_SIZE = 5
DEFAULT_EMBEDDING_DIM = 100
DEFAULT_BATCH_SIZE = 50
DEFAULT_EPOCH_SIZE = 3

############################
# Saving / Loading Helpers #
############################

DIRNAME = os.path.dirname(__file__)
DEFAULT_SAVE_DIR = os.path.join(DIRNAME, 'data/saved_skip_gram')

def is_valid_model_save(path):
    # Check model files exist 
    model_files = (
        f'{path}/model_trained.pt',
        f'{path}/model_config.json',
        f'{path}/vocab_to_int.npy'
    )

    for f in model_files:
        if not os.path.isfile(f):
            return False
    return True

def make_model(config, vocab_to_int):
    return SkipGram(
        len(vocab_to_int),
        embedding_size=config['embedding_diminsions']
    )

def save_model(model, config, vocab_to_int, save_dir=DEFAULT_SAVE_DIR, name=None, time_stamp=True):
    if name is None:
        if time_stamp:
            name = f'SkipGram {datetime.now()}'
        else:
            name = 'SkipGram'
    
    save_path = os.path.join(save_dir, name)
    save_path = gen_postfix(save_path)

    # Create model directory
    os.makedirs(save_path)

    # Save model & config
    torch.save(model.state_dict(), f'{save_path}/model_trained.pt')
    with open(f'{save_path}/model_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Save vocab for translation
    np.save(os.path.join(save_path, 'vocab_to_int.npy'), vocab_to_int)

def load_model(save_path):
    assert is_valid_model_save(save_path)

    # Load config
    # Load config
    config = None
    with open(f'{save_path}/model_config.json', 'r') as f:
        config = json.load(f)

    # Load dict
    vocab_to_int = np.load(f'{save_path}/vocab_to_int.npy', allow_pickle=True).item()

    model = make_model(config, vocab_to_int)
    model.load_state_dict(torch.load(f'{save_path}/model_trained.pt'))


class SkipGram(nn.Module):

    def __init__(self, vocab_size, embedding_size=DEFAULT_EMBEDDING_DIM):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # Hidden layer
        self.emedding = nn.Embedding(self.vocab_size, self.embedding_size)

        # Output
        self.linear = nn.Linear(self.embedding_size, self.vocab_size)
        self.activation = nn.LogSoftmax(1) # Preform softmax over the 1st feature deminson
    
    def forward(self, x, verbose=False):
        out = self.emedding(x) # Embed one-hot encoded input
        if verbose:
            print('Embeddings:')
            print(out)
            print(out.shape)
        out = self.linear(out) # Map embedding to vocabulary
        if verbose:
            print('Linear:')
            print(out)
            print(out.shape)
        out = self.activation(out) # Run y through activation function

        return out

def score_eval_batch(model, batch, criterion, device):
    losses = []
    correct_count, total_count = 0, 0

    model.eval()
    with torch.no_grad():
        words, targets = batch

        # Move embedding layers to proper type and location
        words, targets = words.type(torch.LongTensor), targets.type(torch.LongTensor)
        words, targets = words.to(device), targets.to(device)

        # Send inputs through model
        output = model(words, verbose=False)

        # Calc loss
        loss = criterion(output, targets)
        losses.append(loss.item())

        # Round to get predictions
        # Get the most likely context word from each input
        pred = torch.argmax(output, dim=1)
        correct = pred.eq(targets.view_as(pred))
        
        correct_count += np.sum(correct.to('cpu').numpy())
        total_count += targets.size(0)
    
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
    
    for words, targets in tqdm(loader):
        batch_score = score_eval_batch(model, (words, targets), criterion, device)

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
