# coding: utf-8


import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from sklearn.metrics import f1_score


class BaseLSTM(Chain):
    def __init__(self, n_layers=2, emb_size=60, n_units=60, dropout=0,
                 n_lemma_vocab=None, lemma_emb_size=50, lemma_embed_initial=None,
                 n_pos_vocab=None, pos_emb_size=4,
                 n_dep_vocab=None, dep_emb_size=5,
                 n_dir_vocab=None, dir_emb_size=1):
        super(BaseLSTM, self).__init__()
        with self.init_scope():
            self.dropout = dropout
            self.n_layers = n_layers
            self.n_units = n_units

            # PathをエンコードするNStepLSTM
            self.nslstm = L.NStepLSTM(n_layers=n_layers, in_size=emb_size,
                                      out_size=n_units, dropout=dropout)

            # 各要素のEmbeddingレイヤー
            self.lemma_embed = L.EmbedID(in_size=n_lemma_vocab, out_size=lemma_emb_size, initialW=lemma_embed_initial)
            self.pos_embed = L.EmbedID(n_pos_vocab, pos_emb_size)
            self.dep_embed = L.EmbedID(n_dep_vocab, dep_emb_size)
            self.dir_embed = L.EmbedID(n_dir_vocab, dir_emb_size)

    def __call__(self, hx, cx, xs):
        hy, cy, ys = self.nslstm(hx, cx, xs)
        return hy, cy, ys


def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs


class Path_Encoder(Chain):
    def __init__(self, lstm):
        super(Path_Encoder, self).__init__()
        with self.init_scope():
            self.lstm = lstm
            self.dropout = lstm.dropout
            self.n_layers = lstm.n_layers
            self.n_units = lstm.n_units

    def __call__(self, hx, cx, lemmas, poses, deps, dirs, counts):
        ls_f = sequence_embed(self.lstm.lemma_embed, lemmas)
        ps_f = sequence_embed(self.lstm.pos_embed, poses)
        ds_f = sequence_embed(self.lstm.dep_embed, deps)
        dirs_f = sequence_embed(self.lstm.dir_embed, dirs)

        paths_f = [F.concat((ls_f[i], ps_f[i], ds_f[i], dirs_f[i]), axis=1) for i in range(len(lemmas))]
        hy, cy, ys = self.lstm(hx, cx, paths_f)

        hy = F.scale(hy[self.n_layers - 1], counts, axis=0)
        hy = F.sum(hy, axis=0) / F.sum(counts).data

        last_hidden = hy
        return last_hidden


def word_dropout(id_list, dropout_rate):
    dropouted_id_list = [np.random.choice([i, 0], size=1, p=[1 - dropout_rate, dropout_rate])[0]
                         for i in id_list]
    return dropouted_id_list


class LexNET(Chain):
    def __init__(self, path_encoder, class_n,
                 n_w_vocab=None, w_emb_size=50, embed_initial=None,
                 dropout=0):
        super(LexNET, self).__init__()
        with self.init_scope():
            self.path_encoder = path_encoder
            self.n_units = path_encoder.n_units

            self.concat_w_embedding = L.EmbedID(n_w_vocab, w_emb_size, initialW=embed_initial)

            self.fl1 = L.Linear(None, class_n)
            self.dropout = dropout

    def __call__(self, w1, w2, paths_list):
        w1v = self.concat_w_embedding(w1)
        w2v = self.concat_w_embedding(w2)

        path_reps = []
        for paths in paths_list:
            hx = None
            cx = None
            lemma_seq = [word_dropout(i[0][0], self.dropout) for i in paths]
            pos_seq = [word_dropout(i[0][1], self.dropout) for i in paths]
            dep_seq = [word_dropout(i[0][2], self.dropout) for i in paths]
            dir_seq = [word_dropout(i[0][3], self.dropout) for i in paths]
            count_seq = [i[1] for i in paths]

            lemma_seq = [self.xp.array(l, dtype='i') for l in lemma_seq]
            pos_seq = [self.xp.array(p, dtype='i') for p in pos_seq]
            dep_seq = [self.xp.array(d, dtype='i') for d in dep_seq]
            dir_seq = [self.xp.array(d, dtype='i') for d in dir_seq]
            count_seq = F.reshape(self.xp.array(count_seq, dtype='f'), shape=(-1, 1))

            path_vector = self.path_encoder(hx, cx,
                                            lemma_seq, pos_seq, dep_seq, dir_seq, count_seq)
            path_reps.append(F.reshape(path_vector, shape=(-1, self.n_units)))

        path_reps = F.concat(path_reps, axis=0)
        path_reps = F.reshape(path_reps, shape=(-1, self.n_units))
        rep = F.concat([w1v, path_reps, w2v], axis=1)
        y = self.fl1(rep)
        return y

    def predict(self, w1, w2, paths_list):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            w1v = self.concat_w_embedding(w1)
            w2v = self.concat_w_embedding(w2)

            path_reps = []
            for paths in paths_list:
                hx = None
                cx = None
                lemma_seq = [i[0][0] for i in paths]
                pos_seq = [i[0][1] for i in paths]
                dep_seq = [i[0][2] for i in paths]
                dir_seq = [i[0][3] for i in paths]
                count_seq = [i[1] for i in paths]

                lemma_seq = [self.xp.array(l, dtype='i') for l in lemma_seq]
                pos_seq = [self.xp.array(p, dtype='i') for p in pos_seq]
                dep_seq = [self.xp.array(d, dtype='i') for d in dep_seq]
                dir_seq = [self.xp.array(d, dtype='i') for d in dir_seq]
                count_seq = F.reshape(self.xp.array(count_seq, dtype='f'), shape=(-1, 1))

                path_vector = self.path_encoder(hx, cx,
                                                lemma_seq, pos_seq, dep_seq, dir_seq, count_seq)
                path_reps.append(F.reshape(path_vector, shape=(-1, self.n_units)))

            path_reps = F.concat(path_reps, axis=0)
            path_reps = F.reshape(path_reps, shape=(-1, self.n_units))
            rep = F.concat([w1v, path_reps, w2v], axis=1)
            predicts = self.fl1(rep)

        predicts = F.argmax(predicts, axis=1)
        return chainer.cuda.to_cpu(predicts.data)

    def evaluate(self, w1, w2, paths_list, labels):
        n_test = len(w1)
        predicts = []
        for i in range(0, n_test, 100):
            c_w1 = w1[i:i + 100]
            c_w2 = w2[i:i + 100]
            c_path = paths_list[i:i + 100]
            c_predicts = self.predict(c_w1, c_w2, c_path)
            predicts.append(c_predicts)
        predicts = np.concatenate(predicts)
        score = f1_score(y_pred=predicts, y_true=labels, average='weighted')
        return score


class LexNET_h(Chain):
    def __init__(self, path_encoder, class_n,
                 n_w_vocab=None, w_emb_size=50, embed_initial=None,
                 dropout=0):
        super(LexNET_h, self).__init__()
        with self.init_scope():
            self.path_encoder = path_encoder
            self.n_units = path_encoder.n_units

            self.concat_w_embedding = L.EmbedID(n_w_vocab, w_emb_size, initialW=embed_initial)

            self.fl1 = L.Linear(None, path_encoder.n_units)
            self.fl2 = L.Linear(None, class_n)

            self.dropout = dropout

    def __call__(self, w1, w2, paths_list):
        w1v = self.concat_w_embedding(w1)
        w2v = self.concat_w_embedding(w2)

        path_reps = []
        for paths in paths_list:
            hx = None
            cx = None
            lemma_seq = [word_dropout(i[0][0], self.dropout) for i in paths]
            pos_seq = [word_dropout(i[0][1], self.dropout) for i in paths]
            dep_seq = [word_dropout(i[0][2], self.dropout) for i in paths]
            dir_seq = [word_dropout(i[0][3], self.dropout) for i in paths]
            count_seq = [i[1] for i in paths]

            lemma_seq = [self.xp.array(l, dtype='i') for l in lemma_seq]
            pos_seq = [self.xp.array(p, dtype='i') for p in pos_seq]
            dep_seq = [self.xp.array(d, dtype='i') for d in dep_seq]
            dir_seq = [self.xp.array(d, dtype='i') for d in dir_seq]
            count_seq = F.reshape(self.xp.array(count_seq, dtype='f'), shape=(-1, 1))

            path_vector = self.path_encoder(hx, cx,
                                            lemma_seq, pos_seq, dep_seq, dir_seq, count_seq)
            path_reps.append(F.reshape(path_vector, shape=(-1, self.n_units)))

        path_reps = F.concat(path_reps, axis=0)
        path_reps = F.reshape(path_reps, shape=(-1, self.n_units))
        rep = F.concat([w1v, path_reps, w2v], axis=1)
        h = F.tanh(self.fl1(rep))
        y = self.fl2(h)
        return y

    def predict(self, w1, w2, paths_list):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            w1v = self.concat_w_embedding(w1)
            w2v = self.concat_w_embedding(w2)

            path_reps = []
            for paths in paths_list:
                hx = None
                cx = None
                lemma_seq = [i[0][0] for i in paths]
                pos_seq = [i[0][1] for i in paths]
                dep_seq = [i[0][2] for i in paths]
                dir_seq = [i[0][3] for i in paths]
                count_seq = [i[1] for i in paths]

                lemma_seq = [self.xp.array(l, dtype='i') for l in lemma_seq]
                pos_seq = [self.xp.array(p, dtype='i') for p in pos_seq]
                dep_seq = [self.xp.array(d, dtype='i') for d in dep_seq]
                dir_seq = [self.xp.array(d, dtype='i') for d in dir_seq]
                count_seq = F.reshape(self.xp.array(count_seq, dtype='f'), shape=(-1, 1))

                path_vector = self.path_encoder(hx, cx,
                                                lemma_seq, pos_seq, dep_seq, dir_seq, count_seq)
                path_reps.append(F.reshape(path_vector, shape=(-1, self.n_units)))

            path_reps = F.concat(path_reps, axis=0)
            path_reps = F.reshape(path_reps, shape=(-1, self.n_units))
            rep = F.concat([w1v, path_reps, w2v], axis=1)
            h = F.tanh(self.fl1(rep))
            predicts = self.fl2(h)

        predicts = F.argmax(predicts, axis=1)
        return chainer.cuda.to_cpu(predicts.data)

    def evaluate(self, w1, w2, paths_list, labels):
        n_test = len(w1)
        predicts = []
        for i in range(0, n_test, 100):
            c_w1 = w1[i:i + 100]
            c_w2 = w2[i:i + 100]
            c_path = paths_list[i:i + 100]
            c_predicts = self.predict(c_w1, c_w2, c_path)
            predicts.append(c_predicts)
        predicts = np.concatenate(predicts)
        score = f1_score(y_pred=predicts, y_true=labels, average='weighted')
        return score


class Path_Based(Chain):
    def __init__(self, path_encoder, class_n, dropout=0):
        super(Path_Based, self).__init__()
        with self.init_scope():
            self.path_encoder = path_encoder
            self.n_units = path_encoder.n_units
            self.fl1 = L.Linear(None, class_n)
            self.dropout = dropout

    def __call__(self, paths_list):
        path_reps = []
        for paths in paths_list:
            hx = None
            cx = None
            lemma_seq = [word_dropout(i[0][0], self.dropout) for i in paths]
            pos_seq = [word_dropout(i[0][1], self.dropout) for i in paths]
            dep_seq = [word_dropout(i[0][2], self.dropout) for i in paths]
            dir_seq = [word_dropout(i[0][3], self.dropout) for i in paths]
            count_seq = [i[1] for i in paths]

            lemma_seq = [self.xp.array(l, dtype='i') for l in lemma_seq]
            pos_seq = [self.xp.array(p, dtype='i') for p in pos_seq]
            dep_seq = [self.xp.array(d, dtype='i') for d in dep_seq]
            dir_seq = [self.xp.array(d, dtype='i') for d in dir_seq]
            count_seq = F.reshape(self.xp.array(count_seq, dtype='f'), shape=(-1, 1))

            path_vector = self.path_encoder(hx, cx,
                                            lemma_seq, pos_seq, dep_seq, dir_seq, count_seq)
            path_reps.append(F.reshape(path_vector, shape=(-1, self.n_units)))

        path_reps = F.concat(path_reps, axis=0)
        rep = F.reshape(path_reps, shape=(-1, self.n_units))
        y = self.fl1(rep)
        return y

    def predict(self, paths_list):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            path_reps = []
            for paths in paths_list:
                hx = None
                cx = None
                lemma_seq = [i[0][0] for i in paths]
                pos_seq = [i[0][1] for i in paths]
                dep_seq = [i[0][2] for i in paths]
                dir_seq = [i[0][3] for i in paths]
                count_seq = [i[1] for i in paths]

                lemma_seq = [self.xp.array(l, dtype='i') for l in lemma_seq]
                pos_seq = [self.xp.array(p, dtype='i') for p in pos_seq]
                dep_seq = [self.xp.array(d, dtype='i') for d in dep_seq]
                dir_seq = [self.xp.array(d, dtype='i') for d in dir_seq]
                count_seq = F.reshape(self.xp.array(count_seq, dtype='f'), shape=(-1, 1))

                path_vector = self.path_encoder(hx, cx,
                                                lemma_seq, pos_seq, dep_seq, dir_seq, count_seq)
                path_reps.append(F.reshape(path_vector, shape=(-1, self.n_units)))

            path_reps = F.concat(path_reps, axis=0)
            rep = F.reshape(path_reps, shape=(-1, self.n_units))
            predicts = self.fl1(rep)

        predicts = F.argmax(predicts, axis=1)
        return chainer.cuda.to_cpu(predicts.data)

    def evaluate(self, paths_list, labels):
        # predicts = self.predict(paths_list)
        # score = f1_score(y_pred=predicts, y_true=labels, average='weighted')
        n_test = len(labels)
        predicts = []
        for i in range(0, n_test, 100):
            c_path = paths_list[i:i + 100]
            c_predicts = self.predict(c_path)
            predicts.append(c_predicts)
        predicts = np.concatenate(predicts)
        score = f1_score(y_pred=predicts, y_true=labels, average='weighted')
        return score


class LexNET_Rep(Chain):
    def __init__(self, path_encoder, w2p,
                 n_class, n_w_vocab, emb_size,
                 embed_initial=None, dropout=0):
        super(LexNET_Rep, self).__init__()
        with self.init_scope():
            self.path_encoder = path_encoder

            self.w2p = w2p
            self.w2p.disable_update()

            self.fully_connect = L.Linear(n_class)

            self.embed = L.EmbedID(n_w_vocab, emb_size, embed_initial)

            self.dropout = dropout
            self.n_units = path_encoder.n_units

    def __call__(self, w1, w2, paths_list):
        concat_h = self.w2p.feature_extract(w1, w2)

        w1v = self.embed(w1)
        w2v = self.embed(w2)

        path_reps = []
        for paths in paths_list:
            hx = None
            cx = None
            lemma_seq = [word_dropout(i[0][0], self.dropout) for i in paths]
            pos_seq = [word_dropout(i[0][1], self.dropout) for i in paths]
            dep_seq = [word_dropout(i[0][2], self.dropout) for i in paths]
            dir_seq = [word_dropout(i[0][3], self.dropout) for i in paths]
            count_seq = [i[1] for i in paths]

            lemma_seq = [self.xp.array(l, dtype='i') for l in lemma_seq]
            pos_seq = [self.xp.array(p, dtype='i') for p in pos_seq]
            dep_seq = [self.xp.array(d, dtype='i') for d in dep_seq]
            dir_seq = [self.xp.array(d, dtype='i') for d in dir_seq]
            count_seq = F.reshape(self.xp.array(count_seq, dtype='f'), shape=(-1, 1))

            path_vector = self.path_encoder(hx, cx,
                                            lemma_seq, pos_seq, dep_seq, dir_seq, count_seq)
            path_reps.append(F.reshape(path_vector, shape=(-1, self.n_units)))

        path_reps = F.concat(path_reps, axis=0)
        path_reps = F.reshape(path_reps, shape=(-1, self.n_units))
        rep = F.concat([w1v, path_reps, concat_h, w2v], axis=1)
        y = self.fully_connect(rep)
        return y

    def predict(self, w1, w2, paths_list):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            concat_h = self.w2p.feature_extract(w1, w2)

            w1v = self.embed(w1)
            w2v = self.embed(w2)

            path_reps = []
            for paths in paths_list:
                hx = None
                cx = None
                lemma_seq = [word_dropout(i[0][0], self.dropout) for i in paths]
                pos_seq = [word_dropout(i[0][1], self.dropout) for i in paths]
                dep_seq = [word_dropout(i[0][2], self.dropout) for i in paths]
                dir_seq = [word_dropout(i[0][3], self.dropout) for i in paths]
                count_seq = [i[1] for i in paths]

                lemma_seq = [self.xp.array(l, dtype='i') for l in lemma_seq]
                pos_seq = [self.xp.array(p, dtype='i') for p in pos_seq]
                dep_seq = [self.xp.array(d, dtype='i') for d in dep_seq]
                dir_seq = [self.xp.array(d, dtype='i') for d in dir_seq]
                count_seq = F.reshape(self.xp.array(count_seq, dtype='f'), shape=(-1, 1))

                path_vector = self.path_encoder(hx, cx,
                                                lemma_seq, pos_seq, dep_seq, dir_seq, count_seq)
                path_reps.append(F.reshape(path_vector, shape=(-1, self.n_units)))

            path_reps = F.concat(path_reps, axis=0)
            path_reps = F.reshape(path_reps, shape=(-1, self.n_units))
            rep = F.concat([w1v, path_reps, concat_h, w2v], axis=1)
            predicts = self.fully_connect(rep)

        predicts = F.argmax(predicts, axis=1)
        return chainer.cuda.to_cpu(predicts.data)

    def evaluate(self, w1, w2, paths_list, labels):
        n_test = len(w1)
        predicts = []
        for i in range(0, n_test, 100):
            c_w1 = w1[i:i + 100]
            c_w2 = w2[i:i + 100]
            c_path = paths_list[i:i + 100]
            c_predicts = self.predict(c_w1, c_w2, c_path)
            predicts.append(c_predicts)
        predicts = np.concatenate(predicts)
        score = f1_score(y_pred=predicts, y_true=labels, average='weighted')
        return score


class Classifier(Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__()
        with self.init_scope():
            self.predictor = predictor

    def __call__(self, w1, w2, path_list, labels):
        y = self.predictor(w1, w2, path_list)
        loss = F.softmax_cross_entropy(y, labels)
        return loss


class Classifier_Path_Based(Chain):
    def __init__(self, predictor):
        super(Classifier_Path_Based, self).__init__()
        with self.init_scope():
            self.predictor = predictor

    def __call__(self, path_list, labels):
        y = self.predictor(path_list)
        loss = F.softmax_cross_entropy(y, labels)
        return loss
