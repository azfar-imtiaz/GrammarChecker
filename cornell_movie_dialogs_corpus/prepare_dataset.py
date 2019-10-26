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
                if len(t.split()) > 15 or len(t.split()) < 2:
                    continue
                m = re.search(r'\b(a|an|the)\b', t)
                if m:
                    sents.append(t)
                # sents.append(t)
            line = rfile.readline()

    return sents


def create_seq_mapping(lines, amount, apply_article_removal=False, apply_verb_cont_removal=False):
    rg_article = r'\b(a|an|the)\b'
    rg_verb_contraction = r'(?<=\w)\'[a-z]{1,2}\b'
    random.shuffle(lines)
    altered_lines = []
    for s in lines:
        if apply_article_removal:
            # alt_s = re.sub(rg_article, '', s, count=1)
            # find all articles in the sentence
            matches = list(re.finditer(rg_article, s))
            if len(matches) > 0:
                # select one at random
                random_match = random.choice(matches)
                # remove this randomly selected article from the sentence
                alt_s = s[:random_match.start()] + s[random_match.end():]
                altered_lines.append((alt_s, s))
        elif apply_verb_cont_removal:
            # find all verb contraction in the sentence
            matches = list(re.finditer(rg_verb_contraction, s))
            if len(matches) > 0:
                # select one at random
                random_match = random.choice(matches)
                # remove this randomly selected verb contraction from the sentence
                alt_s = s[:random_match.start()] + s[random_match.end():]
                altered_lines.append((alt_s, s))
        else:
            altered_lines.append((s, s))

        if len(altered_lines) >= amount:
            break
    return altered_lines


if __name__ == '__main__':
    filename = "cornell movie-dialogs corpus/movie_lines.txt"
    sents = read_data_from_file(filename)

    amount_perturbations_artical_removal = int(0.25 * len(sents))
    # random_sents = random.choices(sents, k=amount_perturbations_artical_removal)
    sents_artical_removal = create_seq_mapping(sents, amount_perturbations_artical_removal, apply_article_removal=True, apply_verb_cont_removal=False)

    amount_perturbations_verb_contraction_removal = int(0.25 * len(sents))
    # random_sents = random.choices(sents, k=amount_perturbations_verb_contraction_removal)
    sents_verb_cont_removal = create_seq_mapping(sents, amount_perturbations_verb_contraction_removal, apply_article_removal=False, apply_verb_cont_removal=True)

    amount_correct = int(0.2 * len(sents))
    # random_sents = random.choices(sents, k=amount_correct)
    correct_sents = create_seq_mapping(sents, amount_correct)

    dataset = sents_artical_removal + correct_sents + sents_verb_cont_removal
    random.shuffle(dataset)

    for alt_s in dataset[:10]:
        print(alt_s)

    joblib.dump(dataset, "mapped_seqs.pkl")
