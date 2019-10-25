import torch.nn as nn
import torch.nn.functional as F
from torch import sum as torch_sum


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, decoder_hidden_state, encoder_outputs):
        '''
                The shape of decoder_hidden_state is (1, batch_size, hidden_size), because we're considering one time step
                The shape of encoder_outputs is (max_length, batch_size, hidden_size)

                The shape of decoder_hidden_state*encoder_outputs = (max_length, batch_size, hidden_size)

                Summing across third dimension of this product returns to us a shape of (max_length, batch_size).
        '''

        # compute attention energies by computing dot product of decoder hidden state and encoder outputs
        attention_energies = torch_sum(
            decoder_hidden_state * encoder_outputs, dim=2)
        # take the transpose to get (batch_size, max_length), because we want to take the softmax across batch_size,
        # so that we take the softmax for every individual instance
        attention_energies = attention_energies.t()
        # taking the transpose across dim 1 means take softmax across columns
        normalized_prob_scores = F.softmax(attention_energies, dim=1)
        # thus, the sum of all elements in a single row in normalized_prob_scores would be 1.0
        normalized_prob_scores = normalized_prob_scores.unsqueeze(1)
        # shape of normalized_prob_scores now is (batch_size, 1, max_length). We need to do this because later,
        # we multiply this attention vector with the encoder outputs, the latter of which is a 3D tensor
        return normalized_prob_scores
