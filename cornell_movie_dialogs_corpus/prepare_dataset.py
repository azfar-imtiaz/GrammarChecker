import re
import random
import joblib
from nltk.tokenize import sent_tokenize


def read_data_from_file(filename):
    sents = []
    with open(filename, 'r', errors='ignore') as rfile:
        line = rfile.readline()
        while line:
            t = line.split("+++$+++")[-1].strip()
            # texts = sent_tokenize(text)
            # for t in texts:
            if len(t.split()) > 20 or len(t.split()) < 2:
                line = rfile.readline()
                continue
            m = re.search(r'\b(a|an|the)\b', t)
            n = re.search(r'(?<=\w)\'[a-z]{1,2}\b', t)
            if m or n:
                sents.append(t)
            # sents.append(t)
            line = rfile.readline()

    return sents


def create_seq_mapping(lines, amount_artical_removal, amount_verb_cont_removal, amount_correct):
    rg_article = r'\b(a|an|the)\b'
    rg_verb_contraction = r'(?<=\w)\'[a-z]{1,2}\b'
    random.shuffle(lines)
    altered_lines_article_removal = []
    altered_lines_verb_cont_removal = []
    altered_lines_correct = []
    for index, s in enumerate(lines):
        if index % 5000 == 0 and index > 0:
            print("\t{} rows processed!".format(index))

        if len(altered_lines_article_removal) < amount_artical_removal:
            # alt_s = re.sub(rg_article, '', s, count=1)
            # find all articles in the sentence
            matches = list(re.finditer(rg_article, s))
            if len(matches) > 0:
                # select one at random
                random_match = random.choice(matches)
                # remove this randomly selected article from the sentence
                alt_s = s[:random_match.start()] + s[random_match.end():]
                altered_lines_article_removal.append((alt_s, s))

        if len(altered_lines_verb_cont_removal) < amount_verb_cont_removal:
            # find all verb contraction in the sentence
            matches = list(re.finditer(rg_verb_contraction, s))
            if len(matches) > 0:
                # select one at random
                random_match = random.choice(matches)
                # remove this randomly selected verb contraction from the sentence
                alt_s = s[:random_match.start()] + s[random_match.end():]
                altered_lines_verb_cont_removal.append((alt_s, s))

        if len(altered_lines_correct) < amount_correct:
            altered_lines_correct.append((s, s))

        if len(altered_lines_article_removal) >= amount_artical_removal and \
                len(altered_lines_verb_cont_removal) >= amount_verb_cont_removal and \
                len(altered_lines_correct) >= amount_correct:
            break

    return altered_lines_article_removal + altered_lines_verb_cont_removal + altered_lines_correct


if __name__ == '__main__':
    filename = "cornell movie-dialogs corpus/movie_lines.txt"
    print("Reading data from file...")
    sents = read_data_from_file(filename)

    print("Creating sequence mapping...")
    amount_artical_removal = int(0.15 * len(sents))
    amount_verb_cont_removal = int(0.15 * len(sents))
    amount_correct = int(0.1 * len(sents))
    # random_sents = random.choices(sents, k=amount_correct)
    dataset = create_seq_mapping(sents, amount_artical_removal, amount_verb_cont_removal, amount_correct)

    # dataset = sents_artical_removal + correct_sents + sents_verb_cont_removal
    print("Shuffling dataset...")
    random.shuffle(dataset)

    print("Total items in dataset: {}".format(len(dataset)))
    for alt_s in dataset[:10]:
        print(alt_s)

    joblib.dump(dataset, "mapped_seqs.pkl")
