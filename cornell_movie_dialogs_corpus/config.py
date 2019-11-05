mapped_sequences = "mapped_seqs.pkl"
path_to_dataset = "cornell movie-dialogs corpus/movie_lines.txt"
glove_vectors = "glove_vectors_100d.pkl"
encoder_model = "encoder.pkl"
decoder_model = "decoder.pkl"
vocabulary = "vocabulary.pkl"

# hidden size and number of layers in encoder and decoder must be the same!
encoder_hidden_size = 300
decoder_hidden_size = 300
encoder_num_layers = 2
decoder_num_layers = 2
# this must be 100 when using pre-trained embeddings!
embedding_size = 100
batch_size = 200
encoder_lr = 0.0005
decoder_lr = 0.0005

use_pretrained_embedding = False
teacher_forcing_ratio = 1.0
device = "cuda:1"
num_epochs = 150
