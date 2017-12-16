# coding: utf-8
"""
path_idのtripletfileを辞書へ
"""
from collections import defaultdict
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('parsed_id', type=str)
args = parser.parse_args()

def main(parsed_id):
    path_dict = defaultdict(list)
    with open(parsed_id,'r') as f:
        for line in f:
            try:
                x, y, path = map(int, line.strip().split('\t'))
            except:
                continue

            key = (x, y)
            current = path
            path_dict[key].append(current)

    with open('id_tripret.dic.dump', 'wb') as f:
        pickle.dump(path_dict, f)

if __name__ == '__main__':
    main(args.parsed_id)