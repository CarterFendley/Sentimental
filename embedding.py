# Libraries
import os
import torch
import wandb
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

# Local modules
import skip_gram
import imdb_data
from util.devices import get_device

# Configurations
from config import WANDB_API_KEY

DIRNAME = os.path.dirname(__file__)

SPLIT_FRAC = 0.99
PADDING = 400

device = get_device()

wandb.login(key=WANDB_API_KEY)

skip_gram_config = dict(
    epochs = 5,
    context_size = 5,
    embedding_diminsions = 200,
    lr=0.00001,
    batch_size = 2000,
    dataset='IMDB',
    architecture='Embedding'
)  

def skip_gram_pipeline(config):
    with wandb.init(project='skip-gram', config=config):
        config = wandb.config

        model, data, train_loader, test_loader, valid_loader, criterion, optimizer = make(config)
        print(model)

        train(model, train_loader, valid_loader, criterion, optimizer, config)

        print('Saving movel...')
        skip_gram.save_model(model, config.as_dict(), data['vocab_to_int'])

def make(config):
    data = imdb_data.load_data(
        pad_to=PADDING,
        confirmation=False
    )

    # Create skip gram subsets from train_x of return
    x = []
    y = []
    if False:
        for entry in tqdm(data['train_x'], total=len(data['train_x'])):
            entry_length = len(entry)
            for i, word in enumerate(entry):
                # Do not process padding chars
                if word == 0: 
                    continue 

                for delta in range(-config['context_size'], config['context_size']+1):
                    context_i = i + delta

                    # Validate the position
                    if context_i < 0 or context_i >= entry_length or context_i == i:
                        continue
                    
                    x.append(word)
                    y.append(entry[context_i])
        # Save for future use:
        print('Saving word context pairs...')
        np.save(os.path.join(DIRNAME, 'center_words.npy'), x)
        np.save(os.path.join(DIRNAME, 'context_targets.npy'), y)
    else:
        print('Loading pairs from file...')
        x = np.load(os.path.join(DIRNAME, 'center_words.npy'))
        y = np.load(os.path.join(DIRNAME, 'context_targets.npy'))

    # Split into sets
    total_x = len(x)
    
    data['train_x'] = x[0:int(SPLIT_FRAC*total_x)]
    data['train_y'] = y[0:int(SPLIT_FRAC*total_x)]

    # Get the remaining x and y after train
    x = x[int(SPLIT_FRAC*total_x):]
    y = y[int(SPLIT_FRAC*total_x):]

    data['valid_x'] = x[0:int(len(x)*0.5)]
    data['valid_y'] = y[0:int(len(y)*0.5)]

    data['test_x'] = x[int(len(x)*0.5):]
    data['test_y'] = y[int(len(y)*0.5):]

    if False:
        print('Train: ')
        print('\twords:', len(data['train_x']))
        print('\tcontext:', len(data['train_y']))
        print('Test: ')
        print('\twords:', len(data['test_x']))
        print('\tcontext:', len(data['test_y']))


    # Make loaders
    train_loader = make_loader((data['train_x'], data['train_y']), batch_size=config['batch_size'])
    valid_loader = make_loader((data['valid_x'], data['valid_y']), batch_size=config['batch_size'])
    test_loader = make_loader((data['test_x'], data['test_y']), batch_size=config['batch_size'])

    # Make model
    model = skip_gram.make_model(config, data['vocab_to_int'])
    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    return model, data, train_loader, test_loader, valid_loader, criterion, optimizer


def make_loader(data_pairs, batch_size):
    dataset = TensorDataset(
        torch.from_numpy(data_pairs[0]),
        torch.from_numpy(data_pairs[1])
    )

    loader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=4
    )

    return loader

def train(model, loader, valid_loader, criterion, optimizer, config):
    wandb.watch(model, criterion)

    total_batches = len(loader) * config.epochs
    example_ct, batch_ct = 0, 0

    model.train()
    for epoch in range(config.epochs):
        print('\n-------------------------')
        print(f'Starting Epoch: {epoch+1}/{config.epochs}')
        print('-------------------------')

        for words, targets in tqdm(loader):
            example_ct += len(words)
            batch_ct += 1

            # Zero gradients
            model.zero_grad()

            # Move embedding layers to proper type and location
            words, targets = words.type(torch.LongTensor), targets.type(torch.LongTensor)
            words, targets = words.to(device), targets.to(device)

            # Forward pass
            outputs = model(words)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if ((batch_ct + 1) % 25) == 0:
                if False: print('Loging info')
                train_log(loss, example_ct, epoch)

                if ((batch_ct + 1) % 500) == 0:
                    # Flush cause TQDM hates me
                    print('', flush=True)
                    print('Calculating validation...')
                    score = skip_gram.score_eval_set(model, valid_loader, criterion, device)

                    print(f"\tValidation Loss: {score['mean_loss']}")
                    print(f"\tValidation Accuracy: {score['accuracy']}\n", flush=True)

                    wandb.log({
                        'validation_loss': score['mean_loss'],
                        'validation_accuracy': score['accuracy']
                    })

def train_log(loss, example_ct, epoch):
    loss = float(loss)

    # where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")

def test(model, loader, criterion, config):
    score = skip_gram.score_eval_set(model, loader, criterion, device)

    print(f"Test Loss: {score['mean_loss']}")
    print(f"Test Accuracy: {score['accuracy']}")

    wandb.log({
        'test_loss': score['mean_loss'],
        'test_accuracy': score['accuracy']
    })

if __name__ == '__main__':
    skip_gram_pipeline(skip_gram_config)