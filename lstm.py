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