import torch
import torch.nn as nn


class RecurrentNet(nn.Module):
    def __init__(self, seq_len, vocab_size, embedding_size, num_layers, hidden_size, output_size, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab_size, embedding_size)
        self.lstm1 = nn.LSTM(embedding_size, hidden_size,
                             num_layers=self.num_layers, batch_first=True, dropout=dropout)
        # self.lstm2 = nn.LSTM(hidden_size, hidden_size,
        #                      num_layers=self.num_layers)
        # self.lstm3 = nn.LSTM(hidden_size, hidden_size,
        #                      num_layers=self.num_layers)
        self.fc = nn.Linear(hidden_size * seq_len, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sequence, hidden_layer):
        output = self.emb(sequence)
        # hidden_layer = self.init_hidden(len(sequence[0]))
        output, hidden_layer = self.lstm1(output, hidden_layer)
        # output, hidden_layer = self.lstm2(output, hidden_layer)
        # output, hidden_layer = self.lstm3(output, hidden_layer)
        output = output.contiguous().view(-1, self.hidden_size *
                                          len(sequence[0]))
        output = self.fc(output)
        output = self.sigmoid(output)
        return output, hidden_layer

    def init_hidden(self, seq_len):
        # NOTE THAT LSTM REQUIRES TWO HIDDEN STATES
        return (
            torch.zeros(self.num_layers, seq_len, self.hidden_size).float(),
            torch.zeros(self.num_layers, seq_len, self.hidden_size).float()
        )
