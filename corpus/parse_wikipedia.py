# coding: utf-8
"""
We implement the parser with reference to the implementation of the previous study (https://github.com/vered1986/HypeNET/blob/v2/corpus/parse_wikipedia.py).
"""

import spacy

from spacy.en import English
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('in_file', help='the Wikipedia dump file')
parser.add_argument('out_file', help='the output (parsed) file')
args = parser.parse_args()

def main():

    nlp = English()

    in_file = args.in_file
    out_file = args.out_file

    with open(in_file, 'r') as f_in:
        with open(out_file, 'w') as f_out:

            for paragraph in f_in:

                paragraph = paragraph.replace("'''", '').strip()
                if len(paragraph) == 0:
                    continue

                parsed_par = nlp(paragraph)

                for sent in parsed_par.sents:
                    dependency_paths = parse_sentence(sent)
                    if len(dependency_paths) > 0:
                        for path in dependency_paths:
                            path[0] = nlp(path[0])[0].lemma_
                            path[1] = nlp(path[1])[0].lemma_
                        triple = '\n'.join(['\t'.join(path) for path in dependency_paths])
                        f_out.write(triple+'\n')



def parse_sentence(sent):
    indices = [(token, i, i) for i, token in enumerate(sent) if token.tag_[:2] == 'NN' and len(token.string.strip()) > 2]

    term_pairs = [(x[0], y[0]) for x in indices for y in indices if x[2] < y[1]]
    shortest_paths = [shortest_path(x, y) for x, y in term_pairs]
    paths = [path for path in shortest_paths if path is not None]
    paths = [p for path in paths for p in get_satellite_links(*path)]
    cleaned_paths = [clean_path(*path) for path in paths]
    paths = [list(path) for path in cleaned_paths if path is not None]

    return paths


def shortest_path(x, y):

    x_token = x
    y_token = y
    if not isinstance(x_token, spacy.tokens.token.Token):
        x_token = x_token.root
    if not isinstance(y_token, spacy.tokens.token.Token):
        y_token = y_token.root

    hx = heads(x_token)
    hy = heads(y_token)

    i = -1
    for i in range(min(len(hx), len(hy))):
        if hx[i] is not hy[i]:
            break

    if i == -1:
        lch_idx = 0
        if len(hy) > 0:
            lch = hy[lch_idx]
        elif len(hx) > 0:
            lch = hx[lch_idx]
        else:
            lch = None
    elif hx[i] == hy[i]:
        lch_idx = i
        lch = hx[lch_idx]
    else:
        lch_idx = i-1
        lch = hx[lch_idx]

    hx = hx[lch_idx+1:]
    if lch and check_direction(lch, hx, lambda h: h.lefts):
        return None
    hx = hx[::-1]

    hy = hy[lch_idx+1:]
    if lch and check_direction(lch, hy, lambda h: h.rights):
        return None

    return (x, hx, lch, hy, y)


def heads(token):
    t = token
    hs = []
    while t is not t.head:
        t = t.head
        hs.append(t)
    return hs[::-1]


def check_direction(lch, hs, f_dir):
    return any(modifier not in f_dir(head) for head, modifier in zip([lch] + hs[:-1], hs))


def get_satellite_links(x, hx, lch, hy, y):

    paths = [(None, x, hx, lch, hy, y, None)]

    x_lefts = [tok for tok in x.lefts]
    if len(x_lefts) > 0 and x_lefts[0].tag_ != 'PUNCT' and len(x_lefts[0].string.strip()) > 1:
        paths.append((x_lefts[0], x, hx, lch, hy, y, None))

    y_rights = [tok for tok in y.rights]
    if len(y_rights) > 0 and y_rights[0].tag_ != 'PUNCT' and len(y_rights[0].string.strip()) > 1:
        paths.append((None, x, hx, lch, hy, y, y_rights[0]))

    return paths


def edge_to_string(token, is_head=False):

    t = token
    if not isinstance(token, spacy.tokens.token.Token):
        t = token.root

    return '/'.join([token_to_lemma(token), t.pos_, t.dep_ if t.dep_ != '' and not is_head else 'ROOT'])


def argument_to_string(token, edge_name):

    if not isinstance(token, spacy.tokens.token.Token):
        token = token.root

    return '/'.join([edge_name, token.pos_, token.dep_ if token.dep_ != '' else 'ROOT'])


def direction(dir):

    if dir == UP:
        return '>'
    elif dir == DOWN:
        return '<'


def token_to_string(token):

    if not isinstance(token, spacy.tokens.token.Token):
        return ' '.join([t.string.strip().lower() for t in token])
    else:
        return token.string.strip().lower()


def token_to_lemma(token):

    if not isinstance(token, spacy.tokens.token.Token):
        return token_to_string(token)
    else:
        return token.lemma_.strip().lower()


def clean_path(set_x, x, hx, lch, hy, y, set_y):

    set_path_x = []
    set_path_y = []
    lch_lst = []

    if set_x:
        set_path_x = [edge_to_string(set_x) + direction(DOWN)]
    if set_y:
        set_path_y = [direction(UP) + edge_to_string(set_y)]

    if isinstance(x, spacy.tokens.token.Token) and lch == x:
        dir_x = ''
        dir_y = direction(DOWN)

    elif isinstance(y, spacy.tokens.token.Token) and lch == y:
        dir_x = direction(UP)
        dir_y = ''

    else:
        lch_lst = [edge_to_string(lch, is_head=True)] if lch else []
        dir_x = direction(UP)
        dir_y = direction(DOWN)

    len_path = len(hx) + len(hy) + len(set_path_x) + len(set_path_y) + len(lch_lst)

    if len_path <= MAX_PATH_LEN:
        cleaned_path = '_'.join(set_path_x + [argument_to_string(x, 'X') + dir_x] +
                                [edge_to_string(token) + direction(UP) for token in hx] +
                                lch_lst +
                                [direction(DOWN) + edge_to_string(token) for token in hy] +
                                [dir_y + argument_to_string(y, 'Y')] + set_path_y)
        return token_to_string(x), token_to_string(y), cleaned_path
    else:
        return None


MAX_PATH_LEN = 4
UP = 1
DOWN = 2

if __name__ == '__main__':
    main()