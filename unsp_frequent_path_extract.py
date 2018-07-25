# coding: utf-8

from collections import Counter
import pickle
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--frequency_rank', '-fr', type=int, default=30000)
    args = parser.parse_args()

    path_count = Counter()
    with open('corpus/id_triples', 'r') as f:
        for line in f:
            w1, w2, path = line.strip().split('\t')
            path_id = int(path)

            path_count.update(path_id)

    with open('corpus/id_to_path.dump', 'rb') as f:
        id_to_path = pickle.load(f)

    target_path_id = [i[0] for i in path_count.most_common()[:args.frequency_rank]]
    target_path_counts = [path_count[i] for i in target_path_id]
    target_path = [id_to_path[path_id] for path_id in target_path_id]

    with open('unsp_target_path_id.dump', 'wb') as f:
        pickle.dump(target_path_id, f)

    with open('unsp_target_path_count.dump', 'wb') as f:
        pickle.dump(target_path_counts, f)

    with open('unsp_target_path.dump', 'wb') as f:
        pickle.dump(target_path, f)
