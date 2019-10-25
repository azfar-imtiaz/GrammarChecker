import re
import random
import joblib
from nltk.tokenize import sent_tokenize


def read_data_from_file(filename):
    sents = []
    with open(filename, 'r', errors='ignore') as rfile:
        line = rfile.readline()
        while line:
            text = line.split("+++$+++")[-1].strip()
            texts = sent_tokenize(text)
            for t in texts:
                if len(t.split()) > 15:
                    continue
                m = re.search(r'\b(a|an|the)\b', t)
                if m:
                    sents.append(t)
            line = rfile.readline()

    return sents


def create_seq_mapping(sents, apply_perturbation=False):
    altered_sents = []
    for s in sents:
        if apply_perturbation:
            alt_s = re.sub(r'\b(a|an|the)\b', '', s, count=1)
            altered_sents.append((alt_s, s))
        else:
            altered_sents.append((s, s))
    return altered_sents


if __name__ == '__main__':
    filename = "cornell movie-dialogs corpus/movie_lines.txt"
    sents = read_data_from_file(filename)
    amount_perturbations = int(0.25 * len(sents))
    perturbed_sents = random.choices(sents, k=amount_perturbations)
    altered_sents = create_seq_mapping(
        perturbed_sents, apply_perturbation=True)

    amount_correct = int(0.2 * len(sents))
    correct_sents = random.choices(sents, k=amount_correct)
    correct_sents = create_seq_mapping(correct_sents)

    dataset = altered_sents + correct_sents
    random.shuffle(dataset)

    for alt_s in dataset[:10]:
        print(alt_s)

    joblib.dump(dataset, "mapped_seqs.pkl")
