# coding: utf-8

import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--glove_path', '-g', type=str)
args = parser.parse_args()

def main():
    words = []
    vecs = []
    with open(args.glove_path, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            w = line[0]
            vec = line[1:]
            words.append(w)
            vecs.append(vec)
    n_dim = len(vecs[0])
    vecs = np.array(vecs).astype('f')

    words = ['#UNKNOWN#', '#EOS#', 'X', 'Y'] + words

    token = np.random.normal(size=(4, n_dim)).astype('f')

    vecs = np.vstack([token, vecs])
    np.save(args.glove_path+'.npy', vecs)

    glove_index = {w:i for i, w in enumerate(words)}
    with open('glove_index.dump', 'wb') as f:
        pickle.dump(glove_index, f)


if __name__ == '__main__':
    main()







