# coding: utf-8


import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
args = parser.parse_args()


def main(path_file):
    with open('term_to_id.dump', 'rb') as f:
        lemma2id = pickle.load(f)

    with open('path_to_id.dump', 'rb') as f:
        path2id = pickle.load(f)

    with open(path_file, 'r') as f:
        with open(path_file + '_id', 'w') as l:
            for line in f:
                try:
                    x, y, path = line.strip().split('\t')
                except:
                    continue

                x_id, y_id, path_id = lemma2id.get(x, -1), lemma2id.get(y, -1), path2id.get(path, -1)
                if x_id != -1 and y_id != -1 and path_id != -1:
                    l.write(str(x_id) + '\t' + str(y_id) + '\t' + str(path_id) + '\n')

if __name__ == '__main__':
    alp = ['a', 'b', 'c', 'd', 'e', 'f',
           'g', 'h', 'i', 'j', 'k', 'l',
           'm', 'n', 'o', 'p', 'q', 'r',
           's', 't']
    for i in alp:
        main(args.path+'_a{}_parsed'.format(i))
