# coding: utf-8


import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from chainer import cuda
from chainer.functions.loss import black_out
from chainer import link
from chainer.utils import walker_alias
from chainer import variable


class Unsp_Model(Chain):
    def __init__(self, n_lemma_vocab, n_emb_size, hidden_size=100,
                 n_units=100, counts=None, k=15, init_embed=None, dropout=0, freeze=0):
        super(Unsp_Model, self).__init__()
        with self.init_scope():
            self.lemma_embed = L.EmbedID(n_lemma_vocab, n_emb_size, initialW=init_embed)
            if freeze == 1:
                self.lemma_embed.disable_update()

            self.l1 = L.Linear(hidden_size)
            self.l2 = L.Linear(n_units)
            self.path_ns = L.NegativeSampling(n_units, counts, k)

            self.n_units = n_units
            self.n_lemma_vocab = n_lemma_vocab
            self.counts = counts

    def __call__(self, w1s, w2s, paths):
        batch_size = len(w1s)

        w1e = self.lemma_embed(w1s)
        w2e = self.lemma_embed(w2s)

        concat_embedding = F.concat([w1e, w2e], axis=1)

        h = F.tanh(self.l1(concat_embedding))
        h = F.tanh(self.l2(h))

        loss = self.path_ns(h, paths) / batch_size
        return loss

    def predict(self, w1s, w2s):
        path_W = L.Linear(self.n_units, len(self.counts))
        path_W.W = self.path_ns.W

        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            w1e = self.lemma_embed(w1s)
            w2e = self.lemma_embed(w2s)

            concat_embedding = F.concat([w1e, w2e], axis=1)

            h = F.tanh(self.l1(concat_embedding))
            h = F.tanh(self.l2(h))
            predict = path_W(h)
        return chainer.cuda.to_cpu(predict.data)

    def feature_extract(self, w1s, w2s):
        w1e = self.lemma_embed(w1s)
        w2e = self.lemma_embed(w2s)
        concat_embedding = F.concat([w1e, w2e], axis=1)
        h1 = F.tanh(self.l1(concat_embedding))
        h1 = F.tanh(self.l2(h1))

        concat_embedding2 = F.concat([w2e, w1e], axis=1)
        h2 = F.tanh(self.l1(concat_embedding2))
        h2 = F.tanh(self.l2(h2))

        concat_h = F.concat([h1, h2], axis=1)

        return concat_h
