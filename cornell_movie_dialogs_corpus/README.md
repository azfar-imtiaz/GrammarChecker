# Grammar Correction
The aim of this project is to build a Sequence-to-Sequence model which, upon taking a grammatically incorrect sentence, returns the grammatically correct sentence as output.

## Dataset used
The dataset that I used for this project is the Cornell Movie Dialogs Corpus, which can be found at this link: http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html. It contains 220,579 conversational exchanges between 10,292 pairs of movie characters, and for the most part, they are grammatically correct exchanges. For this project, I was not so much interested in the speakers themselves or the exchange of conversations, but instead more in actual dialogs. So I just pulled the dialogs from this dataset and used them for training the model. The model is trained on pairs of dialogs, where in each pair, the first sentence is the incorrect version, and the second sentence is the correct version.

The basic idea for generating the dataset works like this:

- For each type of grammatical error, decide upon a fixed number of sentences to use for that error
- Sample this fixed number of sentences from the dataset for each error type
- Apply perturbation(s) upon this sentence as per the error type
- Set this (now incorrect) sentence as the input sequence, and set the original (correct) sequence as the output.

In addition, I also selected a certain number of sentences from the dataset and used these sentences as both the input and output sequence in the pair. This was done to feed a set of correct sequence -> correct sequence mappings to the model, so that at evaluation time, it can learn to recognize if a given sequence is already correct and therefore it doesn't need to apply any 'corrections' to it.

## Types of grammatical errors
For this project, I experimented with the following types of errors:

|                     Type of error                    | Percentage of error | Amount of sentences |
|:----------------------------------------------------:|:-------------------:|:-------------------:|
| Removal of articles (a, an, the)                     |         0.15        |        20,838       |
| Removal of second part of verb contractions          |         0.15        |        20,838       |
| Inversion of singular nouns to plural and vice versa |         0.15        |        20,838       |
| Correct sentences                                    |         0.1         |        13,892       |

## Generating the sequence mappings
To generate the sequence mappings to constitute as training and evaluation data for the model, I had to artificially generate the incorrect sequences (their types and amounts are mentioned above). 

#### Removal of articles and verb contractions
For removal of articles and removal of the second part of verb contractions, I used regular expressions to identify the sentences where these perturbations can be applied. For both error types, I only applied this perturbation once per sentence, but I chose where to apply this randomly. So for example, if a sentence contains three articles at different locations, I selected which one to remove randomly, instead of always removing the first one, for instance. Additionally, I applied both of these perturbations to a sentence simultaneously if applicable; so for example, "I've never been to a restaurant as fancy as that!" would be changed to "I never been to restaurant as fancy as that!" This was done so that the model can learn to apply multiple corrections to the same sentence if required.

#### Inversion of singular and plural nouns
Converting a randomly selected noun to either its singular or plural counterpart was slightly more complicated - for each sentence, I extracted its Part-Of-Speech tags by using a pre-trained English model from Spacy. This would be followed by a check to see if the sentence contains any singular and/or plural nouns (by looking for the "NN" and "NNS" tags) - if so, any one of them would be selected randomly and converted to its singular/plural form. For the conversion from singular to plural and vice versa, I used the `singularize` and `pluralize` functions from a handly Python module called `inflection`.

#### Correct sequences
For the correct sequence pairs, I just pick a sentence and use it as both the input and output sequence without applying any perturbations.

#### Examples of sequence pairs

| Incorrect sequence                                                         | Correct sequence                                                             |
|----------------------------------------------------------------------------|------------------------------------------------------------------------------|
| Shit. It my Galiano.                                                       | Shit. It's my Galiano.                                                       |
| Spock, we on leave. You can call me Jim.                                   | Spock, we're on leave. You can call me Jim.                                  |
| They're trying to make me spend the summer here. I leaving in morning.     | They're trying to make me spend the summer here. I'm leaving in the morning. |
| My sister's coming by to pick me up for brunches. Why don't you come, too? | My sister's coming by to pick me up for brunch. Why don't you come, too?     |
| That my license and registration. I wanna be in compliance.                | That's my license and registration. I wanna be in compliance.                |
| I have work to do at plant.                                                | I have work to do at the plant.                                              |
| I decided I didn't want drink...I beginning to wonder.                     | I decided I didn't want a drink...I'm beginning to wonder.                   |
| You need a glasses?                                                        | You need a glass?                                                            |
| Put your fucking hand up! Don't move.                                      | Put your fucking hands up! Don't move.                                       |
| --he not my husband!                                                       | --he's not my husband!                                                       |

## Designing the Network
Since this problem is essentially a sequence-to-sequence mapping task, that is, the input is a sequence of variable length and the output is also a sequence of variable length, the type of model I'm using for this task is an ecoder-decoder model with attention. Since the output will be very similar to the input in most cases (if not exactly the same as in the case of correct sequences as input), I think the attention mechanism will play an important role in helping the model distinguish which parts of the sentence need correction, and what kind of correction should be applied. The model is described in more detail here.

#### Embeddings
The embedding layer is defined separately, outside of any network. This is done so that the same embedding layer can be used for both the encoder and decoder. This will ensure that both networks get the same representations of all words in the vocabulary.

#### Encoder
The encoder network is basically a bidirectional GRU with 2 layers. The input sequence batch is passed through the embedding layer to get vector representations of all the words in each sentence, and then this batch is packed using the `pack_padded_sequence` function, and fed into the GRU layer. THe output is the unpacked through `pad_packed_sequence`, and the bidirectional outputs are then combined through addition. Both the output and the hidden state from the GRU are returned as part of the forward pass of the GRU.

#### Attention
The attention network comes into play after the input sequence has gone through the encoder and we have the encoder output and the encoder hidden state, and when we are passing the output sequence through the decoder. The output from the decoder is computed step by step, and at each step, the dot product of the output from the GRU layer of the decoder (explained in more detail below) and the output from the encoder is computed, which is referred to as attention energies. This is followed by a softmax, which basically helps the model understand which parts of the input sequence to focus on for the output currently being generated by the decoder, as per my understanding. 

#### Decoder
The decoder is perhaps the most complicated network in this configuration. It consists of a unidirectional GRU with 2 layers, two linear layers, and the attention network (since the attention model is actually initialized inside the decoder). 
The sequence is fed to the decoder in a word-by-word fashion. At each timestep, the hidden state from the previous timestep and the current input (which is either the target word, or the output from the previous timestep based on whether we're using teacher forcing or not) are fed into the model. Please note that the initial hidden state to the decoder for each sequence is the final hidden state from the encoder. The input is passed through the embedding layer, and then this input embedding and the hidden state are fed into the GRU. The hidden state returned from this GRU becomes the hidden state to be fed into the decoder for the next timestep. 
The output from the GRU is passed through the attention model along with the encoder output, which returns the attention weights for this timestep. Doing a batch matrix multiplication between these attention weights and the encoder output gives us the context vector, which is then used for generating the decoder output as follows: The output from the decoder GRU and the context vector are concatenated and then fed through a linear layer in the decoder, which reduces the size of this vector from `decoder_hidden_size * 2` to `decoder_hidden_size`. This is followed by a tanh activation function, then the second linear layer which reduces the output to the `output_size` (which is basically the vocabulary size), and finally a log softmax to get probabilities. 

#### Loss function and optimizer
The loss function used here is the NLLLoss (when initializing this loss function, the `ignore_index` is set to the padding token, as we don't want to calculate the loss for padded values in the sequence).
Two optimizers are used - one for the encoder and one for the decoder, and both of them are Adam optimziers. Their learning rates are specified in the config file.

## Training the model
#### Preparing the data
The data preparation process is fairly standard by this point. A vocabulary class is used to get all words in the sequences, and create numeric indices for each word. Additionally, there are three other words in the vocabulary: the starting token, the ending token, and the padding token. For each sentence pair, the ending token is added to the end of each input sentence. This is so that the model can learn where a sentence ends. Similarly when generating the output from the decoder, the first output to the decoder is the starting token. 
The numeric representations of all sentences are generated using the vocabulary's word-to-index mapping, and then these sequences are padded to the maximum sentence length. Additionally, the maximum sentence length for the output sequences is noted, as this is later used when generating the output from the decoder.
For batching, I did not use a dataloader this time; I used a simple for loop with a step size of the batch size to generate the batches. Because of this, the data is manually shuffled before each epoch.

#### Training process?

#### Teacher forcing

## Evaluating the model
#### BLEU score

#### Creating a chat service for human evaluation

## Results
#### Training Loss & Validation Data (for different ratios of teacher forcing)

#### Interesting points

## Final thoughts