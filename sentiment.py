#!/usr/bin/env python3
import os
import time
from skip_gram import SkipGram
from lstm import SentimentLSTM
from imdb_data import load_data
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn # TODO: Needed?
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

import wandb
from config import WANDB_API_KEY

dev = 'cpu'

if torch.cuda.is_available():
    dev = 'cuda'

device = torch.device(dev)


wandb.login(key=WANDB_API_KEY)

skip_gram_config = dict(
    epochs = 3,
    context_size = 5,
    embedding_diminsions = 100,
    batch_size = 50,
    dataset='IMDB',
    architecture='Embedding'
)  

lstm_config = dict(
    epochs = 3,
    batch_size = 50,
    lr = 0.001,
    seq_len = 400,
    output_size = 1,
    embedding_dim = 400,
    hiddem_dim = 500,
    n_layers = 2, # TODO: What does this do?
    clip_grad = 5, # TODO: read more
    dataset='IMDB',
    architecture='LSTM'
)

def lstm_pipeline(hyperparameters):
    with wandb.init(project='lstm', config=hyperparameters):
        config = wandb.config

        model, train_loader, test_loader, valid_loader, criterion, optimizer = make(config)
        print(model)

        train(model, train_loader, criterion, optimizer, config)

        print('Saving model...')
        local_save(model)

        test(model, test_loader, criterion, config)

        return model

def make(config):
    data = load_data(
        pad_to=config['seq_len']
    )

    train_loader = make_loader((data['train_x'], data['train_y']), batch_size=config['batch_size'])
    valid_loader = make_loader((data['valid_x'], data['valid_y']), batch_size=config['batch_size'])
    test_loader = make_loader((data['test_x'], data['test_y']), batch_size=config['batch_size'])

    vocab_size = len(data['vocab_to_int'])

    model = SentimentLSTM(
        vocab_size,
        config['output_size'],
        config['embedding_dim'],
        config['hiddem_dim'],
        config['n_layers']
    )
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    return model, train_loader, test_loader, valid_loader, criterion, optimizer

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

def train(model, loader, criterion, optimizer, config):
    wandb.watch(model, criterion, log='all')

    total_batches = len(loader) * config.epochs
    example_ct = 0
    batch_ct = 0

    model.train()
    for epoch in tqdm(range(config.epochs)):
        h = model.init_hidden(config.batch_size, device)
        
        for reviews, labels in loader:
        #reviews, labels = next(iter(loader))
        #while True:
            if False: print('Starting bach')
            example_ct += len(reviews)
            batch_ct += 1


            if False: print('Reseting hidden vars')
            # TODO: Really no idea what this does
            h = tuple([each.data for each in h])


            if False: print('Zeroing grad')
            # zero gradients
            model.zero_grad()

            if False: print('Converting tensors')
            # Embedding layer needs ints
            reviews = reviews.type(torch.LongTensor) 

            if False: print('Loading tensors to gpu')
            reviews, labels = reviews.to(device), labels.to(device)
            if False: print('Feeding forward')
            output, h = model(reviews, h)
            
            if False: print('Calcing backstep')
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            optimizer.step()

            if ((batch_ct + 1) % 25) == 0:
                if False: print('Loging info')
                train_log(loss, example_ct, epoch)

def train_log(loss, example_ct, epoch):
    loss = float(loss)

    # where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")

def test(model, loader, criterion, config):
    model.eval()

    with torch.no_grad():
        test_losses = []
        correct_ct, total_ct = 0, 0

        h = model.init_hidden(config.batch_size, device) # TODO: Again, WTF

        for reviews, labels in loader:
            h = tuple([each.data for each in h])

            # get predicted outputs
            reviews = reviews.type(torch.LongTensor)
            reviews, labels = reviews.to(device), labels.to(device)
            output, h = model(reviews, h)
            
            # calculate loss
            test_loss = criterion(output.squeeze(), labels.float())
            test_losses.append(test_loss.item())
            
            # convert output probabilities to predicted class (0 or 1)
            pred = torch.round(output.squeeze())  # rounds to the nearest integer
            
            # compare predictions to true label
            correct = pred.eq(labels.float().view_as(pred))
            correct = np.squeeze(correct.to('cpu').numpy())

            total_ct += labels.size(0)
            correct_ct += np.sum(correct)

        print("Test loss: {:.3f}".format(np.mean(test_losses)))
        wandb.log({"test_loss" : "{:.3f}".format(np.mean(test_losses))})

        # accuracy over all test data
        test_acc = correct_ct/total_ct
        print("Test accuracy: {:.3f}".format(test_acc))
        wandb.log({"test_accuracy": "{:.3f}".format(test_acc)})


def local_save(model, save_dir='data/model_saves'):
    save_path = os.path.join(
        os.path.dirname(__file__),
        save_dir
    )

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    file_name = f'Model {datetime.now()}.pt'
    file_path = os.path.join(save_path, file_name)
    torch.save(model, file_path)



if __name__ == '__main__':
    model = lstm_pipeline(lstm_config)