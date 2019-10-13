import random
from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer

import config
from load_data import read_data_from_csv


def dummy_placeholder_func(doc):
    return doc


def shuffle_data(sents_tokenized_train, labels_train):
    zipped_data = list(zip(sents_tokenized_train, labels_train))
    random.shuffle(zipped_data)
    sents_tokenized_train, labels_train = zip(*zipped_data)
    return sents_tokenized_train, labels_train


def equalize_class_data(sents_tokenized_train, labels_train):
    class_1_threshold = 3000
    train_data_eq = []
    train_labels_eq = []

    class_1_count = 0
    for sent, label in zip(sents_tokenized_train, labels_train):
        if label == 0:
            train_data_eq.append(sent)
            train_labels_eq.append(label)
        elif label == 1:
            if class_1_count <= class_1_threshold:
                train_data_eq.append(sent)
                train_labels_eq.append(label)
                class_1_count += 1

    return train_data_eq, train_labels_eq


if __name__ == '__main__':
    sents_tokenized_train, labels_train = read_data_from_csv(
        config.cola_tokenized_tsv_filename_train)
    sents_tokenized_dev, labels_dev = read_data_from_csv(
        config.cola_tokenized_tsv_filename_dev)

    # equalize class counts
    sents_tokenized_train, labels_train = equalize_class_data(
        sents_tokenized_train, labels_train)

    # shuffle the data
    sents_tokenized_train, labels_train = shuffle_data(
        sents_tokenized_train, labels_train)

    cv = CountVectorizer(analyzer='word', tokenizer=dummy_placeholder_func,
                         preprocessor=dummy_placeholder_func, token_pattern=None, ngram_range=(1, 3))
    cvX_train = cv.fit_transform(sents_tokenized_train)
    cvX_dev = cv.transform(sents_tokenized_dev)

    clf = SVC(kernel='rbf', gamma='scale')
    clf.fit(cvX_train, labels_train)
    predictions = clf.predict(cvX_dev)

    print(classification_report(labels_dev, predictions))
