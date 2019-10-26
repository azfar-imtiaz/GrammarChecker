import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, embedding, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.dropout = dropout

        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, dropout=(
            0.0 if self.num_layers == 1 else self.dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden_state=None):
        # pass input sents through embedding layer
        output = self.embedding(input_seq)
        # pack sequence
        packed_output = nn.utils.rnn.pack_padded_sequence(output, input_lengths, enforce_sorted=False)
        # pass the packed embedded sequences through GRU layer
        output, hidden_state = self.gru(packed_output, hidden_state)
        # unpack sequence
        unpacked_output, _ = nn.utils.rnn.pad_packed_sequence(output)
        # sum bidirectional outputs
        output = unpacked_output[:, :, :self.hidden_size] + \
            unpacked_output[:, :, self.hidden_size:]
        return output, hidden_state
