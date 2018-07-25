# coding: utf-8


import argparse
import os
import pickle
import numpy as np

from supervised_model_common import *


def pickle_load(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_prefix', type=str)
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--lr', '-l', type=float, default=0.001)
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--use_cudnn', '-c', type=int, default=0,
                        help='Use CuDNN if the value is 1')
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.mkdir(args.out)

    if args.use_cudnn == 0:
        chainer.global_config.use_cudnn = 'never'


    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        xp = cuda.cupy
    else:
        xp = np


    print('Data reading...')
    with open(args.data_prefix + '/train_data.dump', 'rb') as f:
        train_data = pickle.load(f)
    train_w1s, train_w2s, train_paths, train_labels = train_data
    train_paths = np.array(train_paths)

    with open(args.data_prefix + '/test_data.dump', 'rb') as f:
        test_data = pickle.load(f)
    test_w1s, test_w2s, test_paths, test_labels = test_data

    with open(args.data_prefix + '/val_data.dump', 'rb') as f:
        val_data = pickle.load(f)
    val_w1s, val_w2s, val_paths, val_labels = val_data

    with open(args.data_prefix + '/relations.txt', 'r') as f:
        lines = f.read().strip().split('\n')
        classes = {line.split('\t')[0]: int(line.split('\t')[1]) for line in lines}

    n_classes = len(classes)
    train_labels = np.array([classes[i] for i in train_labels])
    test_labels = np.array([classes[i] for i in test_labels])
    val_labels = np.array([classes[i] for i in val_labels])

    val_w1s = xp.array(val_w1s, dtype=xp.int32)
    val_w2s = xp.array(val_w2s, dtype=xp.int32)
    val_paths = list(val_paths)

    test_w1s = xp.array(test_w1s, dtype=xp.int32)
    test_w2s = xp.array(test_w2s, dtype=xp.int32)
    test_paths = list(test_paths)

    print('Data are read!')

    print('Model building...')
    lemma_index = pickle_load('work/glove_index.dump')
    pos_index = pickle_load('work/pos_index.dump')
    dep_index = pickle_load('work/dep_index.dump')
    dir_index = pickle_load('work/dir_index.dump')

    lemma_embed = np.load('work/glove50.npy')

    n_lemma = len(lemma_index)
    n_pos = len(pos_index)
    n_dep = len(dep_index)
    n_dir = len(dir_index)

    max_val_score = 0
    dropout_rate = [0.0, 0.2, 0.4]
    n_layers = [2]
    f = open(args.out + '/log.txt', 'w')
    f.close()
    val_f = open(args.out + '/val_log.txt', 'w')
    val_f.close()
    test_f = open(args.out + '/test_score.txt', 'w')
    test_f.close()
    test_score = 0

    for layer_num in n_layers:
        for d_r in dropout_rate:

            lstm = BaseLSTM(n_layers=layer_num, emb_size=60, n_units=60, dropout=0,
                            n_lemma_vocab=n_lemma, lemma_emb_size=50, lemma_embed_initial=lemma_embed,
                            n_pos_vocab=n_pos, pos_emb_size=4,
                            n_dep_vocab=n_dep, dep_emb_size=5,
                            n_dir_vocab=n_dir, dir_emb_size=1
                            )
            path_encoder = Path_Encoder(lstm)
            path_based = Path_Based(path_encoder, class_n=n_classes, dropout=d_r)

            model = Classifier_Path_Based(path_based)
            if args.gpu >= 0:
                chainer.cuda.get_device_from_id(args.gpu).use()
                model.to_gpu()

            optimizer = optimizers.Adam(args.lr)
            optimizer.setup(model)

            n_train = len(train_w1s)

            test_score = 0
            c_val = 0
            c_max_val_score = 0
            e = 0

            while c_val <= 7:
                perm = np.random.permutation(n_train)
                for i in range(0, n_train, args.batchsize):
                    c_w1s = xp.array(train_w1s[perm[i:i + args.batchsize]], dtype=xp.int32)
                    c_w2s = xp.array(train_w2s[perm[i:i + args.batchsize]], dtype=xp.int32)
                    c_paths = train_paths[perm[i:i + args.batchsize]]
                    c_labels = xp.array(train_labels[perm[i:i + args.batchsize]], dtype=xp.int32)

                    loss = model(c_paths, c_labels)

                    optimizer.target.cleargrads()
                    loss.backward()
                    optimizer.update()

                    cur_result = '# epoch = {}, minibatch = {}/{}, loss = {}'.format(e + 1,
                                                                                     int(i / args.batchsize) + 1,
                                                                                     int(n_train / args.batchsize) + 1,
                                                                                     loss.data
                                                                                     )
                    with open(args.out + '/log.txt', 'a') as f:
                        f.write('dropout: {} n_layer: {},'.format(str(d_r), str(layer_num)) + cur_result + '\n')

                current_val_score = path_based.evaluate(val_paths, val_labels)

                if current_val_score > c_max_val_score:
                    c_val = 0
                    c_max_val_score = current_val_score
                c_val += 1
                e += 1

                with open(args.out + '/val_log.txt', 'a') as f:
                    f.write('{}\t{}'.format(str(d_r), str(layer_num)) + '\t' + str(current_val_score) + '\n')
                if current_val_score > max_val_score:
                    max_val_score = current_val_score
                    serializers.save_npz(args.out + '/best.model', path_based)
                    test_score = path_based.evaluate(test_paths, test_labels)
                    with open(args.out + '/test_score.txt', 'a') as f:
                        f.write('dropout: {}, n_layers: {}\ttest_score: {}\n'.format(str(d_r), str(layer_num),
                                                                                     str(test_score)))
