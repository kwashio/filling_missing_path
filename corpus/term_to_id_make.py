# coding: utf-8
"""
lemma_indexとは別に、term_to_idを作成し、
tripret_dicとdata_proccesingに用いる。
"""

import pickle

alp = ['a', 'b', 'c', 'd', 'e', 'f',
       'g', 'h', 'i', 'j', 'k', 'l',
       'm', 'n', 'o', 'p', 'q', 'r',
       's', 't']
terms = set()
for i in alp:
    with open('wiki_a{}_parsed.out'.format(i), 'r') as f:
        for line in f:
            w1, w2, path = line.strip().split('\t')
            terms.add(w1)
            terms.add(w2)

term_to_id = {w:i for (i, w) in enumerate(terms)}

with open('term_to_id.dump', 'wb') as f:
    pickle.dump(term_to_id,f)