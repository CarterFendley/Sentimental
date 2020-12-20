import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MultiLabelBinarizer

from tqdm import tqdm
from util import to_one_hot, to_multi_hot

DEFAULT_CONTEXT_SIZE = 5
DEFAULT_EMBEDDING_DIM = 100
DEFAULT_BATCH_SIZE = 50
DEFAULT_EPOCH_SIZE = 3

dev = 'cpu'

if torch.cuda.is_available():
    dev = 'cuda'

DEFAULT_DEVICE = torch.device(dev)

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
        self.activation = nn.LogSoftmax(1) # Preform softmax over the 1st feature deminson
    
    def forward(self, x):
        out = self.emedding(x) # Embed one-hot encoded input
        out = self.linear(out) # Map embedding to vocabulary
        out = self.activation(out) # Run y through activation function

        return out
    
def train_skip_gram(model, tokenized_corpus, exclude_tokens=None, batch_size=DEFAULT_BATCH_SIZE, epochs=DEFAULT_EPOCH_SIZE, device=DEFAULT_DEVICE, verbose=False):
    # Create word context pairs
    x = []
    y = []

    if verbose:
        print('Generating dataset from context')

    if False:
        for entry in tqdm(tokenized_corpus, total=len(tokenized_corpus)):
            entry_length = len(entry)
            for pos, center_word in enumerate(entry):
                if exclude_tokens is not None and center_word in exclude_tokens:
                    continue # Skip excluded

                # Iterate over context positions (inclusive bounds)
                context = []
                for delta in range(-model.context_size, model.context_size +1):
                    context_pos = pos + delta

                    # Make sure pos valid and not zero delta
                    if context_pos < 0 or context_pos >= entry_length or context_pos == pos:
                        continue

                    # Add new pair
                    x.append(center_word)
                    y.append(entry[context_pos])
                    context.append(entry[context_pos])
    else:
        print('Loading data from file')
        x = np.load('/home/carter/src/TDS-LSTM-Tutorial/center_words.npy')
        y = np.load('/home/carter/src/TDS-LSTM-Tutorial/context_targets.npy')

    def pad_context_vectors(vectors):
        pad_to = 2*model.context_size

        # Create vector of zeros of 2*context_size for each vector
        features = np.zeros((len(vectors), pad_to), dtype=int)


        for i, v in enumerate(vectors):
            l = len(v) # Get current length:

            if l <= pad_to:
                zeroes = list(np.zeros(pad_to - l))
                new = zeroes + v
            elif l > pad_to:
                new = v
            
            features[i,:] = np.array(new) #TODO: What is the i,: syntax
        return features

    if verbose:
        import sys

        print('X:')
        print('\tExample', x[0])
        print('\tlength:', len(x))
        print('\tsize:', sys.getsizeof(x))
        print('Y:')
        print('\tExample', y[0])
        print('\tlength:', len(y))
        print('\tsize:', sys.getsizeof(y))

    if True:
        np.save('/home/carter/src/TDS-LSTM-Tutorial/center_words.npy', x)
        np.save('/home/carter/src/TDS-LSTM-Tutorial/context_targets.npy', x)

    data = TensorDataset(
        torch.tensor(x, dtype=torch.long),
        torch.tensor(y, dtype=torch.long)
    )
    data_loader = DataLoader(data, shuffle=True, batch_size=batch_size)

    if verbose:
        print('Training model')

    losses = []
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    if verbose:
        print('Moving model to', device)
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0
        if verbose:
            print('Starting epoch', epoch)
        counter = 0
        
        for word, target in data_loader:
        #word, target = next(iter(data_loader))
        #for i in range(100000):
         
            counter += 1
            print(f'batch: {counter}', sys.getsizeof(word), sys.getsizeof(target), word.size())

            x = word.to(device)
            y_true = target.to(device)

            model.zero_grad()

            y = model(x)
            #print(y)
            #print(y_true)

            loss = loss_function(y, y_true)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print("Epoch: {}/{}...".format(epoch+1, epochs),
                "Loss: {:.6f}...".format(loss.item()))
