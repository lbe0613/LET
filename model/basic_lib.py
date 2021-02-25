#! -*- coding: utf-8 -*-

import mxnet.ndarray as nd
from mxnet.gluon import nn


class TDense(nn.Block):
    def __init__(self, in_dim, out_dim, activation=None, bias=True, dropout=None, **kwargs):
        super(TDense, self).__init__(**kwargs)

        with self.name_scope():
            self.weight = self.params.get('weight', shape=(in_dim, out_dim))
            self.bias = self.params.get('bias', shape=(out_dim,)) if bias is True else None
            self.activation = None if activation is None else nn.Activation(activation)
            self.dropout = None if dropout is None else BayesDropout(dropout['rate'], dropout['axes'])

    def forward(self, x, x_len=None, mode='mul', reshape=None):
        if self.dropout is not None:
            x = self.dropout(x, reshape)

        y = nd.dot(x, self.weight.data(x.context))

        if self.bias is not None:
            y = y + self.bias.data(x.context)

        if self.activation is not None:
            y = self.activation(y)

        if x_len is not None:
            y = Mask(y, x_len, mode)

        return y


class TMLP(nn.Block):
    def __init__(self, dims, dropout=None, layer_norm=False, **kwargs):
        super(TMLP, self).__init__(**kwargs)

        self.dims = dims
        self.layer_norm = layer_norm
        with self.name_scope():
            for i in range(1, len(dims), 1):
                act = None if i == len(dims) - 1 else 'relu'
                setattr(self, f'TDense_{i}', TDense(dims[i - 1], dims[i], activation=act, bias=True, dropout=dropout))

            self.layer_norm = None if layer_norm is False else nn.LayerNorm(scale=True)

    def forward(self, x, x_len=None, mode='mul', reshape=None):
        x_input = x
        for i in range(1, len(self.dims), 1):
            x = getattr(self, f'TDense_{i}')(x, x_len, mode, reshape)

        if self.layer_norm is not None:
            x = self.layer_norm(x_input + x)

        return x


'''
inputs (batch_size, seq_len, input_size)
seq_len (batch_size,)
mode: mul和add，mul used before fully connected layer, set 0；
add used before softmax, set -inf。
'''


def Mask(inputs, seq_len=None, mode='mul'):
    if seq_len is None:
        return inputs

    if mode == 'mul':
        value = 0
    else:
        value = - 1e12

    inputs = nd.swapaxes(inputs, 0, 1)
    inputs_masked = nd.SequenceMask(inputs, sequence_length=seq_len, use_sequence_length=True, value=value)
    inputs_masked = nd.swapaxes(inputs_masked, 0, 1)

    return inputs_masked


class BayesDropout(nn.Block):
    def __init__(self, rate, axes=(), **kwargs):
        super(BayesDropout, self).__init__(**kwargs)

        self.axes = axes
        with self.name_scope():
            self.dropout = nn.Dropout(rate=rate, axes=axes)

    def forward(self, X, reshape=None):
        if reshape is not None:
            shape = X.shape
            reshape = list(reshape)
            reshape[-1] = -1
            X = X.reshape(shape=tuple(reshape))
            X = self.dropout(X)
            X = X.reshape(shape=shape)
        else:
            X = self.dropout(X)

        return X


def bayesian_dropout(fun, X):
    X = X.reshape(shape=(2, -1, X.shape[1], X.shape[2]))
    X = fun(X)
    X = X.reshape(shape=(-1, X.shape[2], X.shape[3]))

    return X


class BIMP(nn.Block):
    def __init__(self, in_dim, num_perspective=20, **kwargs):
        super(BIMP, self).__init__(**kwargs)

        self.num_perspective = num_perspective
        with self.name_scope():
            self.weight = self.params.get('weight', shape=(num_perspective, in_dim))
            self.a = 1

    def forward(self, X):
        X = nd.expand_dims(X, axis=-2)
        reps = [1] * len(X.shape)
        reps[-2] = self.num_perspective
        X = nd.tile(X, reps=tuple(reps))

        W = self.weight.data(X.context)

        delta_dim = len(X.shape) - len(W.shape)
        for i in range(delta_dim):
            W = nd.expand_dims(W, axis=0)

        Y = X * W

        return Y


class BIMPM(nn.Block):
    def __init__(self, in_dim, num_perspective=20, **kwargs):
        super(BIMPM, self).__init__(**kwargs)

        with self.name_scope():
            self.bimp = BIMP(in_dim, num_perspective)

    def forward(self, X1, X2):
        X1 = self.bimp(X1)
        X2 = self.bimp(X2)

        X1_norm = nd.sqrt(nd.sum(X1 * X1, axis=-1) + 1e-12)
        X2_norm = nd.sqrt(nd.sum(X2 * X2, axis=-1) + 1e-12)

        distance_cos = 1 - nd.sum(X1 * X2, axis=-1) / (X1_norm * X2_norm + 1e-12)

        return distance_cos
