# coding: utf-8


import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_prefix', help='the Wikipedia dump file')
args = parser.parse_args()



alp = ['a', 'b', 'c', 'd', 'e', 'f',
       'g', 'h', 'i', 'j', 'k', 'l',
       'm', 'n', 'o', 'p', 'q', 'r',
       's', 't']
terms = set()
for i in alp:
    with open(args.input_prefix+'_a{}_parsed'.format(i), 'r') as f:
        for line in f:
            w1, w2, path = line.strip().split('\t')
            terms.add(w1)
            terms.add(w2)

term_to_id = {w:i for (i, w) in enumerate(terms)}
id_to_term = {i:w for (i, w) in enumerate(terms)}

with open('term_to_id.dump', 'wb') as f:
    pickle.dump(term_to_id,f)

with open('id_to_term.dump', 'wb') as f:
    pickle.dump(id_to_term,f)