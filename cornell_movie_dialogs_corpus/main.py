import torch
import joblib
import torch.nn as nn


import utils
from Encoder import EncoderRNN
from Decoder import DecoderRNN


if __name__ == '__main__':
    dataset = joblib.load('mapped_seqs.pkl')
    vocabulary, sent_pairs = utils.prepare_training_data(dataset[:10])
    input_stuff, output_stuff = utils.generate_training_data(sent_pairs, vocabulary)

    input_tensors, input_lengths = input_stuff
    output_tensors, binary_mask, max_seq_length = output_stuff

    # hidden size and number of layers in encoder and decoder must be the same!
    encoder_hidden_size = 100
    decoder_hidden_size = 100
    decoder_num_layers = 1
    # initialize embedding -> this will be used in both encoder and decoder
    embedding = nn.Embedding(vocabulary.num_words, encoder_hidden_size)

    # initialize the encoder and decoder
    encoder = EncoderRNN(embedding, hidden_size=encoder_hidden_size)
    decoder = DecoderRNN(embedding, hidden_size=decoder_hidden_size,
                         output_size=vocabulary.num_words, num_layers=decoder_num_layers)
    criterion = nn.NLLLoss(ignore_index=vocabulary.PAD_TOKEN)
    # forward pass for encoder
    encoder_output, encoder_hidden = encoder(input_tensors, input_lengths)

    # the starting input for the decoder will always be start_token, for all inputs in the batch
    decoder_input = torch.LongTensor([[vocabulary.START_TOKEN for _ in range(output_tensors.shape[1])]])
    decoder_hidden = encoder_hidden[:decoder_num_layers]
    loss = 0.0
    for i in range(max_seq_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
        # using teacher forcing here
        decoder_input = torch.stack([output_tensors[i, :]])

        target = output_tensors[i]
        mask_loss = criterion(decoder_output, target)
        loss += mask_loss
    print(loss)
    loss.backward()
