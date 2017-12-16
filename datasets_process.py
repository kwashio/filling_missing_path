# coding: utf-8

import pickle
from collections import Counter
import numpy as np
import argparse
import spacy


def edge_decompose(edge):
    direction = ' '
    if edge.startswith('<') or edge.startswith('>'):
        direction = 's' + edge[0]
        edge = edge[1:]
    elif edge.endswith('<') or edge.endswith('>'):
        direction = 'e' + edge[-1]
        edge = edge[:-1]

    try:
        lemma, pos, dep = edge.split('/')
    except:
        return None

    return tuple([lemma, pos, dep, direction])


def edge_tuple2id(edge_tuple,
                  lemma_index, pos_index, dep_index, dir_index):
    lemma = edge_tuple[0]
    pos = edge_tuple[1]
    dep = edge_tuple[2]
    dirc = edge_tuple[3]

    lemma_id = str(lemma_index.get(lemma, 0))
    pos_id = str(pos_index.get(pos, 0))
    dep_id = str(dep_index.get(dep, 0))
    dir_id = str(dir_index.get(dirc, 0))
    return ','.join([lemma_id, pos_id, dep_id, dir_id])


def edges2paths(edges):
    path = [list(map(int, i.split(','))) for i in edges]
    return np.array(path, dtype=np.int32).T.tolist()


def path2indexed(path,
                 lemma_index, pos_index, dep_index, dir_index):
    edges = path.split('_')
    edges = [edge_decompose(i) for i in edges if edge_decompose(i)]
    edges_id = [edge_tuple2id(i, lemma_index, pos_index, dep_index, dir_index)
                for i in edges]
    return edges2paths(edges_id)


def get_paths(id_tripret, id_to_path, x, y):
    x_to_y_path = Counter(id_tripret.get((x, y), []))
    y_to_x_path = Counter(id_tripret.get((y, x), []))

    paths = {id_to_path[path]: count for (path, count) in x_to_y_path.items()}
    paths.update({id_to_path[path].replace('X/', '@@@').replace('Y/', 'X/').replace('@@@', 'Y/'): count
                  for (path, count) in y_to_x_path.items()})
    return paths


def dataset2array(path, id_tripret, id_to_path, term_to_id,
                  lemma_index, pos_index, dep_index, dir_index):
    nlp = spacy.load('en')
    w1_seq = []
    w2_seq = []
    paths_seq = []
    label_seq = []
    with open(path, 'r') as f:
        for line in f:
            w1, w2, label = line.strip().split('\t')
            w1 = nlp(w1)[0].lemma_
            w2 = nlp(w2)[0].lemma_

            w1_id = term_to_id.get(w1, -1)
            w2_id = term_to_id.get(w2, -1)
            paths = get_paths(id_tripret, id_to_path, w1_id, w2_id)
            if paths:
                paths = [(path2indexed(p, lemma_index, pos_index, dep_index, dir_index), c)
                         for (p, c) in paths.items()]
            else:
                paths = [([[0], [0], [0], [0]], 1)]

            w1_seq.append(lemma_index.get(w1, 0))
            w2_seq.append(lemma_index.get(w2, 0))
            paths_seq.append(paths)
            label_seq.append(label)
            print(paths_seq)
    w1_seq = np.array(w1_seq, dtype=np.int32)
    w2_seq = np.array(w2_seq, dtype=np.int32)
    #paths_seq = np.array(paths_seq)
    label_seq = np.array(label_seq)

    return (w1_seq, w2_seq, paths_seq, label_seq)


if __name__ == '__main__':
    with open('work/glove_index.dump', 'rb') as f:
        lemma_index = pickle.load(f)
    with open('work/pos_index.dump', 'rb') as f:
        pos_index = pickle.load(f)
    with open('work/dep_index.dump', 'rb') as f:
        dep_index = pickle.load(f)
    with open('work/dir_index.dump', 'rb') as f:
        dir_index = pickle.load(f)
    with open('corpus/term_to_id.dump', 'rb') as f:
        term_to_id = pickle.load(f)

    with open('corpus/id_to_path.dump', 'rb') as f:
        id_to_path = pickle.load(f)
    with open('corpus/id_tripret.dic.dump', 'rb') as f:
        id_tripret = pickle.load(f)

    for sets in ['/BLESS', '/ROOT09', '/EVALution', '/KHN']:
        path = 'datasets' + sets
        train = path + '/train.tsv'
        test = path + '/test.tsv'
        val = path + '/val.tsv'

        train_data = dataset2array(train, id_tripret, id_to_path, term_to_id,
                                   lemma_index, pos_index, dep_index, dir_index)
        with open(path + '/train_data.dump', 'wb') as f:
            pickle.dump(train_data, f)
        del train_data

        test_data = dataset2array(test, id_tripret, id_to_path, term_to_id,
                                  lemma_index, pos_index, dep_index, dir_index)
        with open(path + '/test_data.dump', 'wb') as f:
            pickle.dump(test_data, f)
        del test_data

        val_data = dataset2array(val, id_tripret, id_to_path, term_to_id,
                                 lemma_index, pos_index, dep_index, dir_index)
        with open(path + '/val_data.dump', 'wb') as f:
            pickle.dump(val_data, f)
