# coding:utf-8

import pickle


def path2id():
    with open('frequent_paths', 'r') as f:
        frequent_paths = set([line.strip() for line in f])

    path_to_id = {p: i for i, p in enumerate(list(frequent_paths))}
    id_to_path = {i: p for i, p in enumerate(list(frequent_paths))}

    del frequent_paths

    with open('path_to_id.dump', 'wb') as f:
        pickle.dump(path_to_id, f)

    with open('id_to_path.dump','wb') as f:
        pickle.dump(id_to_path, f)

if __name__ == '__main__':
    path2id()
