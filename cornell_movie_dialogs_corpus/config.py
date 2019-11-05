mapped_sequences = "mapped_seqs.pkl"
glove_vectors = "glove_vectors_100d.pkl"
encoder_model = "encoder.pkl"
decoder_model = "decoder.pkl"
vocabulary = "vocabulary.pkl"

# hidden size and number of layers in encoder and decoder must be the same!
encoder_hidden_size = 300
decoder_hidden_size = 300
encoder_num_layers = 2
decoder_num_layers = 2
batch_size = 200
encoder_lr = 0.0005
decoder_lr = 0.0005

device = "cuda:1"
num_epochs = 150
