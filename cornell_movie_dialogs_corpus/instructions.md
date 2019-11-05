
# Instructions on running the code

This repository contains a bunch of scripts for different aspects of the project. Here's a rundown on some of the scripts and their functions:

- `prepare_dataset.py` - This script is used for generating the sequence mappings that will be used for training the model, both correct and incorrect. The corpus being used here is the Cornell Movie Dialogs corpus, the path to which is specified in the `config.py` file. This script loads the data from the corpus, introduces grammatical errors in some of the sequences (the amount and nature of these errors has been specified in the report) to create incorrect-correct sequence pairs, as well as some correct-correct sequence pairs, and writes them to a pickle file (the path to which is also specified in the `config.py` file) .
- `main.py` - This script contains the functions to train the seq2seq model, as well as test it on the validation data. It loads the mapped sequences from the path specified in the `config.py` file, trains the model, and saves the model and vocabulary files as specified in the `config.py` file. 
- `chat_service.py` - This script initiates a very basic chat service, which loads the trained models from the specified paths in the `config.py` file, and then initiates a chat service which asks you to enter a sentence, and then returns the grammatically correct version of that sentence. It would be reasonable to send relatively short sentences in dialog form. If your sentence contains a word the model does not know, it will tell you so. You can quit the service by typing 'q' or 'quit'.
- `Encoder.py` - This script contains the class for the encoder model.
- `Decoder.py` - This script contains the class for the decoder model.
- `Attention.py` - This script contains the class for the attention model.
- `utils.py` - This script contains a bunch of utility functions that are used throughout the data generation, model training and model evaluation process.
- `config.py` - This script contains a lot of configuration parameters for data generation, model training and model evaluation. You can specify different file paths here, model parameters such as batch size, hidden layer sizes, learning rates etc., and training parameters such as number of epochs, teacher-forcing ratio, whether to use pre-trained embeddings or not, and the device. Here is some extra information:
 	- `teacher_forcing_ratio` should be a value between 0.0 and 1.0. 0.0 means no teacher forcing at all, 1.0 means teacher forcing all the time, and any value in between will be the ratio with which teacher forcing will be used.
 	- Please note that the embedding size **must** be set to 100 when using pre-trained embeddings.
 	- Please note that if you want to use pre-trained embeddings, you will need the word-to-vector dictionary mapping object that can currently be found at `/home/gusimtmu@GU.GU.SE/GrammarChecker/cornell_movie_dialogs_corpus/glove_vectors_100d.pkl`, and specify the path to it in the `config.py` file under the `glove_vectors` parameter.

To train the model, set all required parameters as per your liking in the `config.py` file, and run the `main.py` script.

Once the models have been trained, you can initiate the chat service by running the `chat_service.py` script. Again, please ensure that the model paths in `config.py` are all correct before running this script!