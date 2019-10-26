import torch
import joblib
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split
# from torch.utils import data

import utils
import config
# from Dataset import Dataset
from Encoder import EncoderRNN
from Decoder import DecoderRNN


def test_model(encoder, decoder, input_elems, output_elems, vocabulary, dev):
    encoder.eval()
    decoder.eval()
    for index in range(0, len(input_elems[1])):
        # get a single element - remember, rows represent words, columns represent individual sentences
        encoder_input = input_elems[0][:, index]
        # reshape the input so that we get a batch size of 1
        encoder_input = encoder_input.view(-1, 1)
        encoder_lengths = [input_elems[1][index]]
        # the first input to the decoder is always the starting token
        decoder_input = torch.LongTensor([[vocabulary.START_TOKEN]])
        decoder_input = decoder_input.to(dev)
        max_seq_length = output_elems[2]
        # forward pass through encoder
        encoder_input = encoder_input.to(dev)
        encoder_output, encoder_hidden = encoder(encoder_input, encoder_lengths)
        encoder_output = encoder_output.to(dev)
        decoder_hidden = encoder_hidden[:decoder.num_layers]
        decoder_hidden = decoder_hidden.to(dev)
        all_words = []
        for i in range(max_seq_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
            output_prob, output_index = torch.max(decoder_output, dim=1)
            output_word = vocabulary.index2word[output_index.item()]
            all_words.append(output_word)
            if output_word == vocabulary.index2word[vocabulary.END_TOKEN]:
                break
            decoder_input = torch.stack([output_index])
        print("Actual text: {}".format(" ".join([vocabulary.index2word[x[0].item()] for x in encoder_input if x[0].item() != vocabulary.PAD_TOKEN and x[0].item() != vocabulary.END_TOKEN])))
        print("Predicted text: {}".format(" ".join(all_words)))


def train_model(encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, input_elems, output_elems, vocabulary, dev, num_epochs=3):
    encoder.train()
    decoder.train()
    loss_values = []
    for epoch in range(num_epochs):
        print("Current epoch: {}".format(epoch + 1))
        epoch_loss = 0.0
        # for input_tensors, input_lengths, output_tensors in training_generator:
        for index in range(0, len(input_elems[1]), config.batch_size):
            start = index
            end = index + config.batch_size
            input_tensors = input_elems[0][:, start: end]
            input_lengths = input_elems[1][start: end]
            output_tensors = output_elems[0][:, start: end]
            max_seq_length = output_elems[2]

            if input_tensors.shape[1] < config.batch_size:
                continue
            # set gradients of optimizers to 0
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # input_tensors, input_lengths = input_elems
            # output_tensors, _, max_seq_length = output_elems

            # forward pass for encoder
            input_tensors = input_tensors.to(dev)
            # input_lengths = input_lengths.to(dev)
            encoder_output, encoder_hidden = encoder(input_tensors, input_lengths)

            # the starting input for the decoder will always be start_token, for all inputs in the batch
            decoder_input = torch.LongTensor([[vocabulary.START_TOKEN for _ in range(output_tensors.shape[1])]])
            decoder_hidden = encoder_hidden[:decoder.num_layers]
            loss = 0.0
            for i in range(max_seq_length):
                decoder_input = decoder_input.to(dev)
                decoder_hidden = decoder_hidden.to(dev)
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
                # using teacher forcing here
                decoder_input = torch.stack([output_tensors[i, :]])

                target = output_tensors[i]
                target = target.to(dev)
                mask_loss = criterion(decoder_output, target)
                loss += mask_loss
            # print("\tLoss: {}".format(loss.item()))
            epoch_loss += loss.item()
            loss.backward()
            # may need to do gradient clipping here
            encoder_optimizer.step()
            decoder_optimizer.step()
        print("Epoch loss: {}".format(epoch_loss))
        loss_values.append(epoch_loss)
    return encoder, decoder, loss_values


if __name__ == '__main__':
    print("Loading data...")
    dataset = joblib.load(config.mapped_sequences)
    print("Generating vocabulary and sentence pairs...")
    vocabulary, sent_pairs = utils.prepare_training_data(dataset)
    dev = torch.device(config.device if torch.cuda.is_available() else "cpu")

    print("Performing train test split...")
    train_sent_pairs, test_sent_pairs = train_test_split(sent_pairs, shuffle=True, test_size=0.2)

    print("Generating training data...")
    input_elems_train, output_elems_train = utils.generate_training_data(train_sent_pairs, vocabulary)

    # initialize embedding -> this will be used in both encoder and decoder
    embedding = nn.Embedding(vocabulary.num_words, config.encoder_hidden_size)

    # initialize the encoder and decoder
    encoder = EncoderRNN(embedding, hidden_size=config.encoder_hidden_size)
    decoder = DecoderRNN(embedding, hidden_size=config.decoder_hidden_size,
                         output_size=vocabulary.num_words, num_layers=config.decoder_num_layers)
    encoder = encoder.to(dev)
    decoder = decoder.to(dev)
    
    criterion = nn.NLLLoss(ignore_index=vocabulary.PAD_TOKEN)
    encoder_optimizer = Adam(encoder.parameters(), lr=config.encoder_lr)
    decoder_optimizer = Adam(decoder.parameters(), lr=config.decoder_lr)

    # train_model(encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, training_generator, max_seq_length)
    print("Training the model...")
    encoder, decoder, loss_values = train_model(encoder, decoder, criterion, encoder_optimizer, decoder_optimizer,
                                   input_elems_train, output_elems_train, vocabulary, dev, num_epochs=config.num_epochs)

    print("Generating testing data...")
    input_elems_test, output_elems_test = utils.generate_training_data(test_sent_pairs, vocabulary)
    
    print("Evaluating the model...")
    test_model(encoder, decoder, input_elems_test, output_elems_test, vocabulary, dev)
