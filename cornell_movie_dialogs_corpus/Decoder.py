import torch.nn as nn
from torch import cat as torch_cat
from torch import tanh as torch_tanh
import torch.nn.functional as F

from Attention import Attention


class DecoderRNN(nn.Module):
    def __init__(self, embedding, hidden_size, output_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, dropout=(
            0.0 if num_layers == 1 else self.dropout), bidirectional=False)
        # this layer is needed when we concatenate the context vector with the output vector of the GRU layer
        self.concatenate = nn.Linear(hidden_size * 2, hidden_size)
        # this layer is needed to reduce the concatenated vector from hidden_size*2 to hidden_size
        self.output = nn.Linear(hidden_size, output_size)
        # we create the object of the attention class here, since it will be used in the decoder forward pass
        self.attention = Attention()

    def forward(self, input_step, prev_hidden_state, encoder_outputs):
        '''
        NOTE: This forward function happens one timestep at a time! Therefore:
        input_step = (1, batch_size) --> single input word fed to the GRU
        prev_hidden_state = (num_layers*num_directions, batch_size, hidden_size) --> final hidden state of encoder
        encoder_outputs = (max_length, batch_size, hidden_size) --> final output state of encoder
        '''
        # this will return shape of (1, batch_size, hidden_size) --> since embedding_size = hidden_size
        output = self.embedding(input_step)
        # output = (1, batch_size, hidden_size); prev_hidden_state = (num_layers*num_directions, batch_size, hidden_size)
        output, prev_hidden_state = self.gru(output, prev_hidden_state)
        # attention_weights = (batch_size, 1, max_length)
        attention_weights = self.attention(output, encoder_outputs)

        '''
        BMM = batch matrix multiplication
        Here we're multiplying the attention weights with the encoder outputs. Before we can do that however,
        we need to transpose the encoder outpust to a shape of (batch_size, max_length, hidden_size).
        This would mean we're doing (batch_size, 1, max_length) * (batch_size, max_length, hidden_size), which
        can work because number of columns in first matrix is equal to number of rows in second matrix: max_length
        This would return a size of (batch_size, 1, hidden_size)
        '''
        context = attention_weights.bmm(encoder_outputs.transpose(0, 1))
        # we do this because we want shape of (batch_size, hidden_size) for concatenation, so we squeeze along dimension 0
        output = output.squeeze(0)
        # we do this because we want shape of (batch_size, hidden_size) for concatenation, so we squeeze along dimension 1
        context = context.squeeze(1)
        # concatenate output of GRU and context vector along dimension 1. This returns (batch_size, hidden_size*2)
        concatenated_output = torch_cat((output, context), 1)
        # reduce (batch_size, hidden_suze*2) to (batch_size, hidden_size)
        concatenated_output = torch_tanh(self.concatenate(concatenated_output))
        # transform from (batch_size, hidden_size) to (batch_size, vocab_size)
        output = self.output(concatenated_output)
        # take softmax across dimension 1 - the columns
        output = F.log_softmax(output, dim=1)
        return output, prev_hidden_state
