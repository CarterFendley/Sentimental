import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from util import to_one_hot, to_multi_hot

DEFAULT_CONTEXT_SIZE = 5
DEFAULT_EMBEDDING_DIM = 100
DEFAULT_BATCH_SIZE = 50
DEFAULT_EPOCH_SIZE = 3

class SkipGram(nn.Module):

    def __init__(self, vocab_size, embedding_size=DEFAULT_EMBEDDING_DIM, context_size=DEFAULT_CONTEXT_SIZE):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.context_size = context_size

        # Hidden layer
        self.emedding = nn.Embedding(self.vocab_size, self.embedding_size)

        # Output
        self.linear = nn.Linear(self.embedding_size, self.vocab_size)
        self.activation = nn.LogSoftmax(self.vocab_size)
    
    def forward(self, x):
        x = self.emedding(x) # Embed one-hot encoded input
        y = self.linear(x) # Map embedding to vocabulary
        out = self.activation(y) # Run y through activation function

        return out
    
    def train(self, tokenized_corpus, batch_size=DEFAULT_BATCH_SIZE, epochs=DEFAULT_EPOCH_SIZE, verbose=False):
        # Create word context pairs
        x = []
        y = []

        if verbose:
            print('Generating dataset from context')

        for entry in tokenized_corpus:
            for pos, center_word in enumerate(entry):
                # Iterate over context positions (inclusive bounds)
                for delta in range(-self.context_size, self.context_size +1):
                    context_pos = pos + delta

                    # Make sure pos valid and not zero delta
                    if context_pos < 0 or context_pos >= len(entry) or context_pos == pos:
                        continue

                    # Add new pair
                    x.append(center_word)
                    y.append(entry[context_pos])

        data = TensorDataset(
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )
        data_loader = DataLoader(data, shuffle=True, batch_size=batch_size)

        if verbose:
            print('Training model')
        losses = []
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001)

        for epoch in range(epochs):
            total_loss = 0
            for word, target in data_loader:
                # Put input and output in one / multi hot form
                x = to_one_hot(word, len(tokenized_corpus))
                x = x.type(torch.LongTensor)

                y_true = to_multi_hot(target, len(tokenized_corpus))
                y_true = y_true.type(torch.LongTensor)

                self.zero_grad()

                y = self(x)

                loss = loss_function(y, y_true)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                print("Epoch: {}/{}...".format(e+1, epochs),
                    "Loss: {:.6f}...".format(loss.item()))
