class Vocabulary:
    def __init__(self):
        self.PAD_TOKEN = 0
        self.START_TOKEN = 1
        self.END_TOKEN = 2

        self.word2index = {}
        self.index2word = {
            self.PAD_TOKEN: "<pad>",
            self.START_TOKEN: "<start>",
            self.END_TOKEN: "<end>"
        }
        self.word2count = {}
        # since we have the three tokens
        self.num_words = 3

    def add_sentence_pair(self, sent_mapping):
        incorrect_sent = sent_mapping[0]
        correct_sent = sent_mapping[1]
        if incorrect_sent == correct_sent:
            # TODO: here maybe instead of split, use NLTK tokenize
            for word in correct_sent.split():
                self.add_word(word)
        else:
            '''
                Theoretically, we can just add the words from correct_sent (since they're a super set of the words)
                in the incorrect_sent. However, later on we may add more types of errors, which may lead to incorrect
                and correct versions of the same sentence having different words, so we add the words from both
                versions here
            '''
            # TODO: here maybe instead of split, use NLTK tokenize
            for word in incorrect_sent.split():
                self.add_word(word)
            # TODO: here maybe instead of split, use NLTK tokenize
            for word in correct_sent.split():
                self.add_word(word)

    def add_word(self, word):
        try:
            self.word2index[word]
            self.word2count[word] += 1
        except KeyError:
            self.word2index[word] = self.num_words
            self.index2word[self.num_words] = word
            self.word2count[word] = 1
            self.num_words += 1

    # function to remove those words from vocabulary having a count less than min_count
    # def trim_vocabulary(self, min_count)
