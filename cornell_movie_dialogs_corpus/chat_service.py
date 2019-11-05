import torch
import joblib

import config
from utils import generate_training_data
from main import test_model


if __name__ == '__main__':
    print("Loading models...")
    encoder = torch.load(config.encoder_model)
    decoder = torch.load(config.decoder_model)
    vocabulary = joblib.load(config.vocabulary)
    if config.use_pretrained_embedding is True:
        glove_vectors = joblib.load(config.glove_vectors)
    else:
        glove_vectors = None
    dev = torch.device(config.device if torch.cuda.is_available() else "cpu")

    print("Chat service initiated!")
    print()

    while True:
        text = input("Please enter a sentence: ")
        text = text.lower().strip()
        if text == 'q' or text == 'quit':
            break

        try:
            input_elems, output_elems = generate_training_data([(text, text)], vocabulary)
        except KeyError as ke:
            print("Oops - seems like I don't know the following word: {}".format(str(ke)))
            continue
        response = test_model(encoder, decoder, input_elems, output_elems, vocabulary, dev,
                              config.use_pretrained_embedding, glove_vectors, chatting=True)
        print("Response: {}".format(response))
        print()
