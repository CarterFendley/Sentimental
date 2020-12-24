#!/usr/bin/env python3
import os
import time
import lstm
from imdb_data import load_data
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn # TODO: Needed?
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from util.devices import get_device

import wandb
from config import WANDB_API_KEY

device = get_device()

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

        model, data, train_loader, test_loader, valid_loader, criterion, optimizer = make(config)
        print(model)

        train(model, train_loader, valid_loader, criterion, optimizer, config)

        print('Saving model...')
        lstm.save_model(model, config.as_dict(), data)

        test(model, test_loader, criterion, config)

        return model

def make(config):
    data = load_data(
        pad_to=config['seq_len']
    )

    train_loader = make_loader((data['train_x'], data['train_y']), batch_size=config['batch_size'])
    valid_loader = make_loader((data['valid_x'], data['valid_y']), batch_size=config['batch_size'])
    test_loader = make_loader((data['test_x'], data['test_y']), batch_size=config['batch_size'])

    model = lstm.make_model(config, data)
    model.to(device)

    criterion = nn.BCELoss()
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
    wandb.watch(model, criterion, log='all')

    total_batches = len(loader) * config.epochs
    example_ct = 0
    batch_ct = 0

    model.train()
    for epoch in range(config.epochs):
        print('\n-------------------------')
        print('Starting Epoch: ', epoch+1)
        print('-------------------------')
        
        for reviews, labels in tqdm(loader):
        #reviews, labels = next(iter(loader))
        #while True:
            if False: print('Starting bach')
            example_ct += len(reviews)
            batch_ct += 1

            if False: print('Zeroing grad')
            # zero gradients
            model.zero_grad()

            if False: print('Converting tensors')
            # Embedding layer needs ints
            reviews = reviews.type(torch.LongTensor) 

            if False: print('Loading tensors to gpu')
            reviews, labels = reviews.to(device), labels.to(device)
            if False: print('Feeding forward')
            output = model(reviews)
            
            if False: print('Calcing backstep')
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            optimizer.step()

            if ((batch_ct + 1) % 25) == 0:
                if False: print('Loging info')
                train_log(loss, example_ct, epoch)

                if ((batch_ct + 1) % 100) == 0:
                    # Flush cause TQDM hates me
                    print('', flush=True)
                    print('Calculating validation...')
                    score = lstm.score_eval_set(model, valid_loader, criterion, device)

                    print(f"\tValidation Loss: {score['mean_loss']}")
                    print(f"\tValidation Accuracy: {score['accuracy']}\n", flush=True)

                    wandb.log({
                        'valid_loss': score['mean_loss'],
                        'valid_accuracy': score['accuracy']
                    })

def train_log(loss, example_ct, epoch):
    loss = float(loss)

    # where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")

def test(model, loader, criterion, config):
    score = lstm.score_eval_set(model, loader, criterion, device)

    print(f"Test Loss: {score['mean_loss']}")
    print(f"Test Accuracy: {score['accuracy']}")

    wandb.log({
        'test_loss': score['mean_loss'],
        'test_accuracy': score['accuracy']
    })


if __name__ == '__main__':
    model = lstm_pipeline(lstm_config)