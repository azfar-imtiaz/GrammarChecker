mapped_sequences = "mapped_seqs.pkl"

# hidden size and number of layers in encoder and decoder must be the same!
encoder_hidden_size = 100
decoder_hidden_size = 100
decoder_num_layers = 2
batch_size = 200
encoder_lr = 0.01
decoder_lr = 0.01

device = "cuda:1"
num_epochs = 50