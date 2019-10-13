import csv


def read_data_from_csv(filename):
    sents_tokenized = []
    labels = []
    with open(filename, 'r') as rfile:
        reader = csv.reader(rfile, delimiter='\t')
        for row in reader:
            sent = row[3]
            sent_tokenized = sent.split()
            label = int(row[1])

            sents_tokenized.append(sent_tokenized)
            labels.append(label)

    return sents_tokenized, labels
