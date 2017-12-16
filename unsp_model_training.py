# coding: utf-8

import pickle
import argparse
import numpy as np
from chainer import optimizers, serializers
from unsp_model_common import *
import os


def pickle_load(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--use_cudnn', '-c', type=int, default=0,
                        help='Use CuDNN if the value is 1')
    parser.add_argument('--out', '-o', type=str)

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

    print('data read../')
    with open(args.data, 'rb') as f:
        w1s, w2s, paths = pickle.load(f)

    counts = pickle_load('unsp_target_path_count.dump')
    target_path_id = pickle_load('unsp_target_path_id.dump')

    path_id2id = {p: i for i, p in enumerate(target_path_id)}
    paths = [path_id2id[i] for i in paths]

    w1s = np.array(w1s, 'i')
    w2s = np.array(w2s, 'i')
    paths = np.array(paths, 'i')
    print('finish!')

    print('model building...')
    lemma_index = pickle_load('work/glove_index.dump')
    lemma_embed = np.load('work/glove50.npy')
    n_lemma = len(lemma_index)

    f = open(args.out + '/log.txt', 'w')
    f.close()

    model = Unsp_Model(n_lemma_vocab=n_lemma, n_emb_size=50, n_units=100,
                       counts=counts, init_embed=lemma_embed, k=5)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = optimizers.Adam(0.001)
    optimizer.setup(model)
    print('finish!')

    n_train = len(w1s)

    for e in range(5):
        perm = np.random.permutation(n_train)
        for i in range(0, n_train, args.batchsize):
            c_w1s = xp.array(w1s[perm[i:i + args.batchsize]], dtype=xp.int32)
            c_w2s = xp.array(w2s[perm[i:i + args.batchsize]], dtype=xp.int32)
            c_paths = xp.array(paths[perm[i:i + args.batchsize]], dtype=xp.int32)

            loss = model(c_w1s, c_w2s, c_paths)

            optimizer.target.cleargrads()
            loss.backward()
            optimizer.update()

            cur_result = '# epoch = {}, minibatch = {}/{}, loss = {}'.format(e + 1,
                                                                             int(i / args.batchsize) + 1,
                                                                             int(n_train / args.batchsize) + 1,
                                                                             loss.data
                                                                             )
            with open(args.out + '/log.txt', 'a') as f:
                f.write(cur_result + '\n')

        serializers.save_npz(args.out + '/unsp_model.model', model)
