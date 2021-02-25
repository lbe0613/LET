#! -*- coding: utf-8 -*-

from basic_lib import *

'''
Fast-DiSA
'''

class FastDiSA(nn.Block):
    def __init__(self, in_dim, nb_head, size_per_head=None, has_mlp=True, rate=0.2, dropout_axes=(0, 2), **kwargs):
        super(FastDiSA, self).__init__(**kwargs)

        self.has_mlp = has_mlp
        if size_per_head is None:
            size_per_head = int(in_dim / nb_head)
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.out_dim = nb_head * size_per_head

        with self.name_scope():
            self.tdense_q = TDense(in_dim, self.out_dim, bias=False)
            self.tdense_k = TDense(in_dim, self.out_dim, bias=False)
            self.tdense_v = TDense(in_dim, self.out_dim, bias=False)
            self.tdense_s2t_1 = TDense(self.out_dim, self.out_dim, activation='relu')
            self.tdense_s2t_2 = TDense(self.out_dim, self.out_dim)
            axes = dropout_axes
            if self.has_mlp is True:
                self.mlp = TMLP((self.out_dim, 2 * self.out_dim, self.out_dim), dropout={'rate': rate, 'axes': axes},
                                layer_norm=False)

            self.dropout_Q = BayesDropout(rate=rate, axes=axes)
            self.dropout_K = BayesDropout(rate=rate, axes=axes)
            self.dropout_V = BayesDropout(rate=rate, axes=axes)
            self.dropout_s2t_1 = BayesDropout(rate=rate, axes=axes)
            self.dropout_s2t_2 = BayesDropout(rate=rate, axes=axes)

    def forward(self, Q, K, V, Q_len=None, V_len=None, Q_reshape=None, graph_matrix=None):
        if Q_reshape is None:
            V_reshape = None
        else:
            V_reshape = V.reshape(shape=(2, -1, V.shape[1], V.shape[2])).shape

        Q = self.dropout_Q(Q, Q_reshape)
        Q = self.tdense_q(Q)
        Q = Q.reshape((-1, Q.shape[1], self.nb_head, self.size_per_head))
        Q = nd.transpose(Q, (0, 2, 1, 3))
        Q = Q.reshape(shape=(-1, Q.shape[2], Q.shape[3]))

        K = self.dropout_K(K, V_reshape)
        K = self.tdense_k(K)
        K = K.reshape((-1, K.shape[1], self.nb_head, self.size_per_head))
        K = nd.transpose(K, (0, 2, 1, 3))
        K = K.reshape(shape=(-1, K.shape[2], K.shape[3]))

        V_H = self.dropout_V(V, V_reshape)
        H_0 = self.tdense_v(V_H)
        H = H_0.reshape((-1, H_0.shape[1], self.nb_head, self.size_per_head))
        H = nd.transpose(H, (0, 2, 1, 3))
        H = H.reshape(shape=(-1, H.shape[2], H.shape[3]))

        # compute t2t score，then mask seq
        A = nd.batch_dot(Q, K, transpose_b=True) / (self.size_per_head ** 0.5)
        A = A.reshape((-1, self.nb_head, A.shape[1], A.shape[2]))
        A = nd.transpose(A, (0, 3, 2, 1))
        A = Mask(A, V_len, mode='add')
        A = nd.transpose(A, (0, 3, 2, 1))
        A = nd.exp(nd.clip(A, a_min=-1e12, a_max=10))
        if graph_matrix is not None:
            A = A * graph_matrix

        attention_weight = A

        A = A.reshape(shape=(-1, A.shape[2], A.shape[3]))

        # compute s2t score
        V_F = self.dropout_s2t_1(V, V_reshape)
        F = self.tdense_s2t_1(V_F, V_len)
        F = self.dropout_s2t_2(F, V_reshape)
        F = self.tdense_s2t_2(F, V_len)
        F = F.reshape((F.shape[0], F.shape[1], self.nb_head, -1))
        F = nd.transpose(F, (0, 2, 1, 3))
        F = F.reshape(shape=(-1, F.shape[2], F.shape[3]))
        F = nd.exp(nd.clip(F, a_min=-1e12, a_max=10))

        # compute output
        E = nd.batch_dot(A, F * H) / (nd.batch_dot(A, F) + 1e-12)
        E = E.reshape(shape=(-1, self.nb_head, E.shape[1], E.shape[2]))
        E = nd.transpose(E, (0, 2, 1, 3))
        E = E.reshape(shape=(E.shape[0], E.shape[1], self.out_dim))

        if self.has_mlp is True:
            O = self.mlp(E, Q_len, 'mul', Q_reshape)
        else:
            O = E

        return O, attention_weight


'''
Fast-DiSA with Interaction (BiMPM)
'''


class FastParallelDiSA(nn.Block):
    def __init__(self, in_dim, nb_head, out_dim=None, **kwargs):
        super(FastParallelDiSA, self).__init__(**kwargs)

        size_per_head = int(in_dim / nb_head)
        self.in_dim = size_per_head * nb_head
        self.out_dim = self.in_dim if out_dim is None else out_dim

        num_pers = 20
        with self.name_scope():
            self.FastDiSA = FastDiSA(self.in_dim, nb_head, has_mlp=False, dropout_axes=(0, 2))
            self.bimpm = BIMPM(self.in_dim, num_pers)
            self.mlp = TMLP((in_dim + num_pers, 2 * self.out_dim, self.out_dim), dropout={'rate': 0.2, 'axes': (0, 2)},
                            layer_norm=False)

    def forward(self, X, X_len, reshape, graph_matrix=None):
        X_list = nd.split(X, axis=0, num_outputs=2)
        X_len_list = nd.split(X_len, axis=0, num_outputs=2)

        Q = X
        K = nd.concat(X_list[1], X_list[0], dim=0)
        V = K
        Q_len = X_len
        V_len = nd.concat(X_len_list[1], X_len_list[0], dim=0)

        Q = nd.concat(X, Q, dim=0)
        K = nd.concat(X, K, dim=0)
        V = nd.concat(X, V, dim=0)
        Q_len = nd.concat(X_len, Q_len, dim=0)
        V_len = nd.concat(X_len, V_len, dim=0)

        reshape_aug = list(reshape)
        reshape_aug[0] = 4

        if graph_matrix is not None:
            graph_matrix = nd.concat(graph_matrix, nd.ones_like(graph_matrix), dim=0)
        C, A = self.FastDiSA(Q, K, V, Q_len, V_len, tuple(reshape_aug), graph_matrix)

        C_list = nd.split(C, axis=0, num_outputs=2)

        Dist_cos = self.bimpm(C_list[0], C_list[1])

        H = nd.concat(C_list[0], Dist_cos, dim=-1)

        Y = self.mlp(H, X_len, 'mul', reshape)

        return Y


'''
Convert a sequence into a fixed vector through multi dim attention
'''


class MultiDimEncodingSentence(nn.Block):
    """docstring for MultiDimEncodingSentence"""

    def __init__(self, in_dim, out_dim=None, rate=0.2, dropout_axes=(0, 2), **kwargs):
        super(MultiDimEncodingSentence, self).__init__(**kwargs)
        if out_dim is None:
            self.out_dim = in_dim
        else:
            self.out_dim = out_dim

        with self.name_scope():
            self.tdense_v = TDense(in_dim, self.out_dim, bias=False)
            self.tdense_s2t_1 = TDense(in_dim, in_dim, bias=True, activation='relu')
            self.tdense_s2t_2_1 = TDense(in_dim, 1, bias=True)
            self.tdense_s2t_2_2 = TDense(in_dim, self.out_dim, bias=True)
            axes = dropout_axes
            self.dropout_H = BayesDropout(rate=rate, axes=axes)
            self.dropout_A = BayesDropout(rate=rate, axes=axes)
            self.dropout_A_1 = BayesDropout(rate=rate, axes=axes)
            self.dropout_A_2 = BayesDropout(rate=rate, axes=axes)

    def forward(self, V, V_len, reshape, seg_mask1=None, seg_mask2=None):

        V_H = self.dropout_H(V, reshape)
        H = self.tdense_v(V_H, V_len, 'mul')

        V_A = self.dropout_A(V, reshape)
        A = self.tdense_s2t_1(V_A)
        A = self.dropout_A_1(A, reshape)
        A_1 = self.tdense_s2t_2_1(A, V_len, 'add')

        V_A_2 = self.dropout_A_2(V, reshape)
        A_2 = self.tdense_s2t_2_2(V_A_2, V_len, 'add')
        A = A_1 + A_2 - nd.mean(A_2, axis=-1, keepdims=True)

        A1 = nd.softmax(A, axis=-2)
        O1 = nd.sum(A1 * H, axis=-2, keepdims=False)

        A = nd.exp(nd.clip(A, a_min=-1e12, a_max=10))
        AH = A * H
        if seg_mask1 is not None:
            O2 = nd.batch_dot(seg_mask1, AH) / (nd.batch_dot(seg_mask1, A) + 1e-12)

        if seg_mask2 is not None:
            O3 = nd.batch_dot(seg_mask2, AH) / (nd.batch_dot(seg_mask2, A) + 1e-12)

        if seg_mask1 is not None and seg_mask2 is not None:
            return O1, O2, O3
        elif seg_mask1 is not None and seg_mask2 is None:
            return O1, O2
        else:
            return O1


'''
Score Layer: Predict the final similarity
'''


class ScoreLayer(nn.Block):
    """docstring for ScoreLayer"""

    def __init__(self, hidden_size=128, nb_class=1, dropout=True, rate=0.2, has_sigmoid=True, **kwargs):
        super(ScoreLayer, self).__init__(**kwargs)

        self.nb_class = nb_class
        self.dropout = dropout
        self.has_sigmoid = has_sigmoid
        with self.name_scope():
            self.dense_1 = nn.Dense(hidden_size, activation='relu', use_bias=True)
            self.dense_2 = nn.Dense(int(hidden_size * 0.5), activation='relu', use_bias=True)
            self.dense_3 = nn.Dense(nb_class, activation=None, use_bias=False)
            if self.dropout is True:
                self.dropout_1 = nn.Dropout(rate=rate)
                self.dropout_2 = nn.Dropout(rate=rate)
                self.dropout_3 = nn.Dropout(rate=rate)

    def forward(self, V, V_len=None):
        if self.dropout is True:
            V = self.dropout_1(V)
        H1 = self.dense_1(V)
        if self.dropout is True:
            H1 = self.dropout_2(H1)

        H2 = self.dense_2(H1)
        if self.dropout is True:
            H2 = self.dropout_3(H2)

        O = self.dense_3(H2)
        return O


'''
Sequence (or Graph) Attention Block without Interaction
'''


class AttentionBlock(nn.Block):
    def __init__(self, in_dim, nb_head, out_dim=None, size_per_head=None, rate=0.2, dropout_axes=(0, 2), pooling=True,
                 **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)

        if size_per_head is None:
            size_per_head = int(in_dim / nb_head)

        self.pooling = pooling
        with self.name_scope():
            setattr(self, 'FastDiSA', eval('FastDiSA')(in_dim, nb_head, rate=rate, dropout_axes=dropout_axes))
            setattr(self, 'layer_norm', nn.LayerNorm(scale=True))
            if self.pooling:
                setattr(self, 'encoder',
                        eval('MultiDimEncodingSentence')(in_dim, out_dim=out_dim, rate=rate, dropout_axes=dropout_axes))

    def forward(self, V, V_len=None, reshape=None, seg_mask=None, graph_matrix=None):
        H, A = getattr(self, 'FastDiSA')(V, V, V, V_len, V_len, reshape, graph_matrix)
        H1 = getattr(self, 'layer_norm')(V + H)

        if self.pooling:
            if seg_mask is None:
                S = getattr(self, 'encoder')(H1, V_len, reshape, seg_mask)
                return S
            else:
                S, H = getattr(self, 'encoder')(H1, V_len, reshape, seg_mask)
                return S, H, H1
        else:
            return H1


'''
Attention Block with Interaction (BIMPM)
'''


class BimpmBlock(nn.Block):
    def __init__(self, in_dim, nb_head, size_per_head=None, dropout_axes=(0, 2), pooling=True, **kwargs):
        super(BimpmBlock, self).__init__(**kwargs)

        if size_per_head is None:
            size_per_head = int(in_dim / nb_head)

        self.pooling = pooling
        with self.name_scope():
            setattr(self, 'FastParallelDiSA', eval('FastParallelDiSA')(in_dim, nb_head))
            setattr(self, 'layer_norm', nn.LayerNorm(scale=True))
            if self.pooling:
                self.encoder = MultiDimEncodingSentence(in_dim, dropout_axes=dropout_axes)

    def forward(self, V, V_len=None, reshape=None, seg_mask=None, graph_matrix=None):
        H = getattr(self, 'FastParallelDiSA')(V, V_len, reshape, graph_matrix)
        H = getattr(self, 'layer_norm')(V + H)

        if self.pooling:
            if seg_mask is None:
                S = getattr(self, 'encoder')(H, V_len, reshape, seg_mask)
                return S
            else:
                S, H = getattr(self, 'encoder')(H, V_len, reshape, seg_mask)
                return S, H
        else:
            return H


class GRU(nn.Block):
    def __init__(self, layer_size, **kwargs):
        super(GRU, self).__init__(**kwargs)

        with self.name_scope():
            self.r = nn.Dense(layer_size, flatten=False)
            self.i = nn.Dense(layer_size, flatten=False)
            self.in_ = nn.Dense(layer_size, flatten=False)
            self.hn_ = nn.Dense(layer_size, flatten=False)

    def forward(self, x, h, seq_len=None):
        data = nd.concat(x, h, dim=-1)
        r = nd.sigmoid(self.r(data))
        i = nd.sigmoid(self.i(data))
        n = nd.tanh(self.in_(x) + r * self.hn_(h))
        h_new = (1 - i) * n + i * h

        O = Mask(h_new, seq_len, 'mul')
        return O
