import re
import spacy
import random
import joblib
import inflection
# from nltk.tokenize import sent_tokenize

import config


def read_data_from_file(filename):
    sents = []
    with open(filename, 'r', errors='ignore') as rfile:
        line = rfile.readline()
        while line:
            t = line.split("+++$+++")[-1].strip()
            # texts = sent_tokenize(text)
            # for t in texts:
            if len(t.split()) > 15 or len(t.split()) < 2:
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


def invert_singular_plural_noun(doc, pos_tags, sent):
    nouns_count = pos_tags.count("NN") + pos_tags.count("NNS")
    rand_noun_index = random.randint(0, nouns_count - 1)
    current_noun_count = 0
    word_to_change = None
    is_plural = False
    for index, word in enumerate(doc):
        # text = word.text
        if word.tag_ == 'NN' or word.tag_ == 'NNS':
            if current_noun_count == rand_noun_index:
                word_to_change = word.text
                if word.tag_ == 'NNS':
                    is_plural = True
                break
            else:
                current_noun_count += 1

    inverted_word = None
    m = re.search(r'(^|\b)%s\b' % word_to_change, sent)
    if m:
        if is_plural:
            inverted_word = inflection.singularize(word_to_change)
        else:
            inverted_word = inflection.pluralize(word_to_change)
        alt_s = re.sub(r'(?<=(^|\b))%s(?=\b)' % word_to_change, inverted_word, sent, count=1)
        return alt_s

    # print(word_to_change)
    # print(sent)
    return None


def create_seq_mapping(lines, amount_artical_removal, amount_verb_cont_removal, amount_singular_plural, amount_correct, nlp=None):
    rg_article = r'\b(a|an|the)\b'
    # rg_verb_contraction = r'(?<=\w)\'[a-z]{1,2}\b'
    rg_verb_contraction = r'(?<=\w)\'[(ve)(s)(d)(m)(re)(ll)]{1,2}(?!\')\b'
    random.shuffle(lines)

    altered_lines_incorrect = []
    altered_lines_correct = []
    amount_incorrect = amount_artical_removal + amount_verb_cont_removal + amount_singular_plural
    article_removed_count = 0
    verb_cont_removed_count = 0
    singular_plural_inverted_count = 0
    for index, s in enumerate(lines):
        if index % 5000 == 0 and index > 0:
            print("\t{} rows processed!".format(index))
        doc = nlp(s)
        pos_tags = [w.tag_ for w in doc]

        # check if sentence contains article or verb contraction or singular/plural noun - if not, skip it
        art_found = bool(re.search(rg_article, s))
        vb_found = bool(re.search(rg_verb_contraction, s))
        pos_tag_found = True if 'NN' in pos_tags or 'NNS' in pos_tags else False

        if art_found is None and vb_found is None and pos_tag_found is False:
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

        elif pos_tag_found and singular_plural_inverted_count < amount_singular_plural:
            alt_s = invert_singular_plural_noun(doc, pos_tags, s)
            if alt_s is not None:
                singular_plural_inverted_count += 1
                altered_lines_incorrect.append((alt_s, s))

        # if either article or verb contraction found and we haven't reached the limit for correct sents
        elif len(altered_lines_correct) < amount_correct:
            altered_lines_correct.append((s, s))

        # if limit is reached for both correct and incorrect sentences, break out of the loop
        if len(altered_lines_correct) >= amount_correct and len(altered_lines_incorrect) >= amount_incorrect:
            break

    print("\nSentences with articles altered: {}".format(article_removed_count))
    print("Sentences with verb contractions altered: {}".format(verb_cont_removed_count))
    print("Sentences with singular/plural nouns inverted: {}".format(singular_plural_inverted_count))
    print("Sentences left unaltered: {}\n".format(len(altered_lines_correct)))
    return altered_lines_correct, altered_lines_incorrect


if __name__ == '__main__':
    filename = "cornell movie-dialogs corpus/movie_lines.txt"
    print("Reading data from file...")
    sents = read_data_from_file(filename)

    print("Loading Spacy English model...")
    nlp = spacy.load('en_core_web_sm')

    print("Creating sequence mapping...")
    # sents = sents[:1000]
    amount_artical_removal = int(0.15 * len(sents))
    amount_verb_cont_removal = int(0.15 * len(sents))
    amount_sing_plural_invertion = int(0.15 * len(sents))
    amount_correct = int(0.1 * len(sents))

    correct_sents, incorrect_sents = create_seq_mapping(sents, amount_artical_removal, amount_verb_cont_removal, amount_sing_plural_invertion, amount_correct, nlp)
    dataset = incorrect_sents + correct_sents

    print("Shuffling dataset...")
    random.shuffle(dataset)

    print("Total items in dataset: {}".format(len(dataset)))
    for alt_s in dataset[:10]:
        print(alt_s)

    joblib.dump(dataset, config.mapped_sequences)
