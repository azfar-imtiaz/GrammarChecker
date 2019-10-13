import torch
from torch.utils import data
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence

import config
from Dataset import Dataset
from RecurrentNet import RecurrentNet
from load_data import read_data_from_csv
from train_svm import equalize_class_data


def create_vocabulary(sents_tokenized):
    tokens = sum(sents_tokenized, [])
    vocabulary = list(set(tokens))
    token_to_int_mapping = {tkn: i + 1 for i, tkn in enumerate(vocabulary)}
    return vocabulary, token_to_int_mapping


def get_sent_numeric_representations(sents_tokenized, vocab_mapping):
    numeric_sents = []
    for sent in sents_tokenized:
        numeric_sent = [vocab_mapping[tkn] for tkn in sent]
        numeric_sents.append(numeric_sent)
    return numeric_sents


if __name__ == '__main__':
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    print("Reading data from csv...")
    sents_tokenized_train, labels_train = read_data_from_csv(
        config.cola_tokenized_tsv_filename_train)
    sents_tokenized_dev, labels_dev = read_data_from_csv(
        config.cola_tokenized_tsv_filename_dev)

    print("Generating vocabulary...")
    vocabulary, token_to_int_mapping = create_vocabulary(
        sents_tokenized_train + sents_tokenized_dev)

    print("Equalizing class counts...")
    sents_tokenized_train, labels_train = equalize_class_data(
        sents_tokenized_train, labels_train)

    # sents_tokenized_train = sents_tokenized_train[:2000]
    print("Generating numeric representations of sentences...")
    sents_numeric_train = get_sent_numeric_representations(
        sents_tokenized_train, token_to_int_mapping)
    sents_numeric_dev = get_sent_numeric_representations(
        sents_tokenized_dev, token_to_int_mapping)

    print("Padding sequences...")
    sents_padded_train = pad_sequence(
        [torch.LongTensor(seq) for seq in sents_numeric_train], batch_first=True)
    # Add a random list of size equal to length of longest sequence in training set, so that length of
    # padded sequences for dev set is same. We will remove this later
    # This needs to be done as we can't specify length in PyTorch's pad_sequence. Mauybe look into Keras
    # functionality of this?
    sents_numeric_dev.append([12] * sents_padded_train.size(1))
    sents_padded_dev = pad_sequence(
        [torch.LongTensor(seq) for seq in sents_numeric_dev], batch_first=True)
    sents_padded_dev = sents_padded_dev[:-1]

    # sents_padded_train = sents_padded_train[:2000]
    print("Total training examples: {}".format(len(sents_padded_train)))

    print("Creating data generator...")
    params = {
        'batch_size': config.BATCH_SIZE,
        'shuffle': True
    }
    training_set = Dataset(sents_padded_train, labels_train)
    training_generator = data.DataLoader(training_set, **params)

    # dev_set = Dataset(sents_padded_dev, labels_dev)
    # testing_generator = data.DataLoader(dev_set, **params)

    print("Initializing model...")
    model = RecurrentNet(sents_padded_train.size(
        1), len(vocabulary) + 1, 300, config.NUM_LAYERS, 500, 2)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = CrossEntropyLoss()

    for epoch in range(config.NUM_EPOCHS):
        print("Epoch number: {}".format(epoch + 1))
        epoch_loss = 0.0
        hidden_layer = model.init_hidden(sents_padded_train.size(1))
        for i, (local_batch, local_labels) in enumerate(training_generator):
            optimizer.zero_grad()
            # hidden_layer = hidden_layer.data
            hidden_layer = tuple([e.data for e in hidden_layer])
            hidden_layer = hidden_layer.to(device)
            local_batch = local_batch.to(device)
            local_labels = local_labels.to(dev)
            output, hidden_layer = model(local_batch, hidden_layer)
            loss = criterion(output, local_labels)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()

        print("Loss for this epoch is: {}".format(epoch_loss))

    model.eval()
    total_predictions, correct_predictions = 0, 0
    for i, (padded_sent, label) in enumerate(zip(sents_padded_dev, labels_dev)):
        total_predictions += 1
        hidden_layer = model.init_hidden(sents_padded_dev.size(1))
        hidden_layer = hidden_layer.to(device)
        input = torch.stack([padded_sent])
        input = input.to(dev)
        pred, _ = model(input, hidden_layer)
        _, prediction = torch.max(pred.data, dim=1)
        if prediction == label:
            correct_predictions += 1
        else:
            print(" ".join(sents_tokenized_dev[i]))

    accuracy = (correct_predictions / total_predictions) * 100
    print("Model accuracy: {}".format(accuracy))
