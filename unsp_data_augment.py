# coding: utf-8

import argparse
import pickle
from unsupervised_model_common import *
import numpy as np
from chainer import serializers
from collections import Counter
import os


def pickle_load(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


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


def path_predict(model, w1s, w2s, k):
    predict = model.predict(w1s, w2s)
    sorted_predict = np.argsort(predict, axis=1)[:, -k:]
    return sorted_predict


def get_path(target_path_id_list, target_path):
    target = Counter(target_path_id_list)
    paths = {target_path[path]: count for (path, count) in target.items()}
    return paths

def data_aug(model, data, k, target_path,
             lemma_index, pos_index, dep_index, dir_index):
    w1s, w2s, path_data, labels = data
    sorted_predict = path_predict(model, w1s, w2s, k)
    for i, predict_ids in enumerate(sorted_predict):
        predict_ids = Counter(predict_ids)
        paths = {target_path[path]:c for (path, c) in predict_ids.items()}
        paths = [(path2indexed(p, lemma_index, pos_index, dep_index, dir_index), c)
                 for (p, c) in paths.items()]
        path_data[i] += paths

    r_sorted_predict = path_predict(model, w2s, w1s, k)
    for i, predict_ids in enumerate(r_sorted_predict):
        predict_ids = Counter(predict_ids)
        paths = {target_path[path].replace('X/', '@@@').replace('Y/', 'X/').replace('@@@', 'Y/'):c for (path, c) in predict_ids.items()}
        paths = [(path2indexed(p, lemma_index, pos_index, dep_index, dir_index), c)
                 for (p, c) in paths.items()]
        path_data[i] += paths
    return (w1s, w2s, path_data, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', '-d', type=str)
    parser.add_argument('--unsp_model', '-u', type=str)
    parser.add_argument('--add_sample', '-k', type=int, default=5)
    parser.add_argument('--output', '-o', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    # word embedding
    with open('work/pos_index.dump', 'rb') as f:
        pos_index = pickle.load(f)
    with open('work/dep_index.dump', 'rb') as f:
        dep_index = pickle.load(f)
    with open('ork/dir_index.dump', 'rb') as f:
        dir_index = pickle.load(f)

    lemma_index = pickle_load('work/glove_index.dump')
    lemma_embed = np.load('work/glove50.npy')
    n_lemma = len(lemma_index)

    # path
    counts = pickle_load('unsp_target_path_count.dump')
    target_path = pickle_load('/home/washio/Unsp_path/work_for_word2path/target_path.dump')

    model = Word2Path(n_lemma_vocab=n_lemma, n_emb_size=50, n_units=100,
                      counts=counts, init_embed=lemma_embed)
    serializers.load_npz(args.unsp_model, model)


   #for sets in ['/BLESS', '/ROOT09', '/EVALution', '/K&H+N']:
    for sets in ['/BLESS', '/ROOT09', '/EVALution']:
        data_path = args.data_folder + sets
        train = data_path + '/train_data.dump'
        test = data_path + '/test_data.dump'
        val = data_path + '/val_data.dump'

        # data augmentation
        with open(train, 'rb') as f:
            train_data = pickle.load(f)
        augmented_train_data = data_aug(model, train_data, args.add_sample, target_path,
                                        lemma_index, pos_index, dep_index, dir_index)

        with open(test, 'rb') as f:
            test_data = pickle.load(f)
        augmented_test_data = data_aug(model, test_data, args.add_sample, target_path,
                                       lemma_index, pos_index, dep_index, dir_index)

        with open(val, 'rb') as f:
            val_data = pickle.load(f)
        augmented_val_data = data_aug(model, val_data, args.add_sample, target_path,
                                      lemma_index, pos_index, dep_index, dir_index)


        # save
        out_path = args.output + sets
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        with open(out_path + '/train_data.dump', 'wb') as f:
            pickle.dump(augmented_train_data, f)

        with open(out_path + '/test_data.dump', 'wb') as f:
            pickle.dump(augmented_test_data, f)

        with open(out_path + '/val_data.dump', 'wb') as f:
            pickle.dump(augmented_val_data, f)




