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


def remove_component_from_sent(rg, sent):
    matches = list(re.finditer(rg, sent))
    if len(matches) > 0:
        random_match = random.choice(matches)
        sent = sent[:random_match.start()] + sent[random_match.end():]
        return sent
    return None


def create_seq_mapping(lines, amount_artical_removal, amount_verb_cont_removal, amount_correct):
    rg_article = r'\b(a|an|the)\b'
    # rg_verb_contraction = r'(?<=\w)\'[a-z]{1,2}\b'
    rg_verb_contraction = r'(?<=\w)\'[(ve)(s)(d)(m)(re)(ll)]{1,2}(?!\')\b'
    random.shuffle(lines)
    altered_lines_incorrect = []
    altered_lines_correct = []
    amount_incorrect = amount_artical_removal + amount_verb_cont_removal
    article_removed_count = 0
    verb_cont_removed_count = 0
    for index, s in enumerate(lines):
        if index % 5000 == 0 and index > 0:
            print("\t{} rows processed!".format(index))

        # first check if sentence contains either an article or verb contraction or both - if not, skip it
        art_found = re.search(rg_article, s)
        vb_found = re.search(rg_verb_contraction, s)
        if art_found is None and vb_found is None:
            continue

        # if both article and verb cont found, and we have not reached the limit for both article removal and
        # verb contraction removal
        if art_found and vb_found \
                and article_removed_count < amount_artical_removal \
                and verb_cont_removed_count < amount_verb_cont_removal:
            alt_s = remove_component_from_sent(rg_article, s)
            alt_s = remove_component_from_sent(rg_verb_contraction, alt_s)
            if alt_s is not None:
                article_removed_count += 1
                verb_cont_removed_count += 1
                altered_lines_incorrect.append((alt_s, s))

        # if article found and we have not reached the limit for article removal
        elif art_found and article_removed_count < amount_artical_removal:
            alt_s = remove_component_from_sent(rg_article, s)
            article_removed_count += 1
            altered_lines_incorrect.append((alt_s, s))

        # if verb contraction found and we have not reached the limit for verb contraction removal
        elif vb_found and verb_cont_removed_count < amount_verb_cont_removal:
            alt_s = remove_component_from_sent(rg_verb_contraction, s)
            verb_cont_removed_count += 1
            altered_lines_incorrect.append((alt_s, s))

        # if either article or verb contraction found and we haven't reached the limit for correct sents
        elif len(altered_lines_correct) < amount_correct:
            altered_lines_correct.append((s, s))

        # if limit is reached for both correct and incorrect sentences, break out of the loop
        if len(altered_lines_correct) >= amount_correct and len(altered_lines_incorrect) >= amount_incorrect:
            break

    return altered_lines_correct, altered_lines_incorrect


if __name__ == '__main__':
    filename = "cornell movie-dialogs corpus/movie_lines.txt"
    print("Reading data from file...")
    sents = read_data_from_file(filename)

    print("Creating sequence mapping...")
    amount_artical_removal = int(0.25 * len(sents))
    amount_verb_cont_removal = int(0.25 * len(sents))
    amount_correct = int(0.2 * len(sents))

    correct_sents, incorrect_sents = create_seq_mapping(sents, amount_artical_removal, amount_verb_cont_removal, amount_correct)
    dataset = incorrect_sents + correct_sents

    print("Shuffling dataset...")
    random.shuffle(dataset)

    print("Total items in dataset: {}".format(len(dataset)))
    for alt_s in dataset[:10]:
        print(alt_s)

    joblib.dump(dataset, "mapped_seqs.pkl")
