# coding: utf-8

import pickle
import argparse

if __name__ == '__main__':

    with open('unsp_target_path_id.dump', 'rb') as f:
        target_path_id = pickle.load(f)

    with open('work/glove_index.dump', 'rb') as f:
        glove_index = pickle.load(f)

    with open('corpus/id_to_term.dump', 'rb') as f:
        id_to_term = pickle.load(f)

    w1_ids = []
    w2_ids = []
    path_ids = []
    with open('corpus/id_triples', 'r') as f:
        for line in f:

            w1, w2, path = line.strip().split('\t')
            w1 = id_to_term[int(w1)]
            w2 = id_to_term[int(w2)]

            if int(path) in target_path_id and w1 in glove_index.keys() and w2 in glove_index.keys():
                w1_ids.append(glove_index[w1])
                w2_ids.append(glove_index[w2])
                path_ids.append(int(path))

    unsp_data = (w1_ids, w2_ids, path_ids)

    with open('unsp_data.dump', 'wb') as f:
        pickle.dump(unsp_data, f)
