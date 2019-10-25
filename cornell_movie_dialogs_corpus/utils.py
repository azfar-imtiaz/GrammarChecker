import re
import torch
import joblib
import itertools
import unicodedata
import torch.nn as nn
from torch.nn import NLLLoss

from Vocabulary import Vocabulary
from Encoder import EncoderRNN
from Decoder import DecoderRNN


# convert unicode string to ASCII, for reference: https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(text):
    return ''.join([char for char in unicodedata.normalize('NFD', text) if unicodedata.category(char) != 'Mn'])


# normalize string by converting to ascii, adding space before any punct mark, and replacing multiple spaces
# with single space
def normalize_string(text):
    text = unicode_to_ascii(text.lower().strip())
    text = re.sub(r'([\.\!\?])', r' \1 ', text)
    # this one might not be needed since we're splitting later on space anyway
    text = re.sub(r'\s+', " ", text)
    return text


# initialize vocabulary, normalize sentence pairs in our dataset
def prepare_training_data(sent_pairs):
    voc = Vocabulary()
    sent_pairs_normalized = []
    for sent_p in sent_pairs:
        # normalize incorrect and correct sentence in pair, and append them to normalized sentence pairs
        incorrect_sent_normalized = normalize_string(sent_p[0])
        correct_sent_normalized = normalize_string(sent_p[1])

        normalized_sents_pair = (
            incorrect_sent_normalized, correct_sent_normalized)
        sent_pairs_normalized.append(normalized_sents_pair)
        # add normalized sentence pair to vocabulary
        voc.add_sentence_pair(normalized_sents_pair)

    return voc, sent_pairs_normalized


# return a setence where each word is replaced with its index, and an END token added to the end of it
def get_numeric_representation_sentence(sent, voc):
    numeric_sent = [voc.word2index[w] for w in sent.split()]
    numeric_sent.append(voc.END_TOKEN)
    return numeric_sent


# make each sentence equal to longest sentence length in the batch, by adding PAD_TOKEN to the end of shorter sents
def pad_sents(batch_sents, fill_value):
    padded_sents = list(itertools.zip_longest(
        *batch_sents, fillvalue=fill_value))
    return padded_sents


# generate a binary mask matrix against the batch of numeric sentences where every index that does not have the pad
# token has a 1. For pad tokens, we have 0 in this mask
def generate_binary_matrix(batch_numeric_sents, fill_value):
    binary_mask = []
    for index, seq in enumerate(batch_numeric_sents):
        temp_row = []
        for token in seq:
            temp_row.append(
                1) if token != fill_value else temp_row.append(fill_value)
        binary_mask.append(temp_row)
    return binary_mask


# get numeric representations and pad all sents in the input batch, convert them to a tensor, and return that along
# with lengths of each sequence. I think this is for the encoder...
def get_padded_sequences_input(sents_batch, voc):
    numeric_sents_batch = [
        get_numeric_representation_sentence(s, voc) for s in sents_batch]
    numeric_sents_lengths = [len(s) for s in numeric_sents_batch]
    padded_sequences = pad_sents(numeric_sents_batch, fill_value=voc.PAD_TOKEN)
    padded_tensors = torch.LongTensor(padded_sequences)
    return padded_tensors, numeric_sents_lengths


# get numeric representations and pad all sents in the output batch, convert them to a tensor, and return that along
# with lengths of each sequence, as well as the binary mask. I think this is for the decoder...
def get_padded_sequences_output(sents_batch, voc):
    numeric_sents_batch = [
        get_numeric_representation_sentence(s, voc) for s in sents_batch]
    max_seq_length = max([len(s) for s in numeric_sents_batch])
    padded_sequences = pad_sents(numeric_sents_batch, fill_value=voc.PAD_TOKEN)
    binary_mask = generate_binary_matrix(
        padded_sequences, fill_value=voc.PAD_TOKEN)

    binary_mask = torch.BoolTensor(binary_mask)
    padded_tensors = torch.LongTensor(padded_sequences)
    return padded_tensors, binary_mask, max_seq_length


# get training data by processing the sentence pairs
def generate_training_data(sent_pairs, voc):
    sent_pairs = sorted(sent_pairs, key=lambda x: len(x[0].split()), reverse=True)
    input_sents, output_sents = [], []
    for sent_p in sent_pairs:
        input_sents.append(sent_p[0])
        output_sents.append(sent_p[1])
    input_tensors, input_lengths = get_padded_sequences_input(
        input_sents, voc)
    output_tensors, binary_mask, max_seq_length = get_padded_sequences_output(
        output_sents, voc)
    return (input_tensors, input_lengths), (output_tensors, binary_mask, max_seq_length)
