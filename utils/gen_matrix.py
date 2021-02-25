# -*- coding: UTF-8 -*-

import numpy as np

def gen_reachable_matrix(neighbor_matrix):
    dim = neighbor_matrix.shape[0]
    P = neighbor_matrix
    for i in range(2, dim+1):
        A = P.dot(neighbor_matrix)
        A_bool = (A > 0).astype(int)
        if (P == A_bool).all():
            break
        else:
            P = A_bool
    return P


def gen_edge_matrix(edge, dim, edge_mode):
    matrix = np.identity(dim, dtype=np.int)

    for e in edge:
        if e[0] < dim and e[1] < dim:
            matrix[e[0]][e[1]] = 1

    forward_edge = matrix
    backward_edge = matrix.T
    bi_ad_edge = [forward_edge, backward_edge]
    if edge_mode == 'reachable':
        forward_re_edge = gen_reachable_matrix(forward_edge)
        backward_re_edge = gen_reachable_matrix(backward_edge)

        bi_re_edge = [forward_re_edge, backward_re_edge]
        return forward_edge, bi_re_edge
    else:
        return forward_edge, bi_ad_edge


def gen_convert_map(sent1_len, sent2_len, single_sent_len_max, sent_len_max):
    mat1 = np.zeros((single_sent_len_max, sent_len_max), dtype=np.int)
    mat2 = np.zeros((single_sent_len_max, sent_len_max), dtype=np.int)
    for i in range(sent1_len):
        mat1[i][i] = 1
    for i in range(sent2_len):
        mat2[i][sent1_len + i - 1] = 1
    return mat1, mat2


def gen_lattice_map(lattice, lattice_len, sent_len):
    pos_1 = np.zeros((lattice_len,), dtype=np.int)
    pos_2 = np.zeros((lattice_len,), dtype=np.int)
    mat = np.zeros((lattice_len, sent_len), dtype=np.int)
    forward_position = np.zeros(lattice_len)
    backward_position = np.zeros(lattice_len)
    max_pos = 0
    for i, index in enumerate(lattice):
        s = index[1]
        e = index[2]
        pos_1[i] = s
        pos_2[i] = e
        forward_position[i] = s
        backward_position[i] = e
        max_pos = e if e > max_pos else max_pos
        for j in range(s, e + 1):
            mat[i][j] = 1

    backward_position = max_pos - backward_position
    return mat, forward_position, backward_position, pos_1, pos_2


def gen_sense_map(sememe_mat, sense_list, lattice_len_max, sense_len_max, sememe_len_max, use_default_sense):
    sememe_mat_new = np.zeros((sense_len_max, sememe_len_max), dtype=np.int)
    shape_raw = sememe_mat.shape
    sememe_mat_new[:shape_raw[0], :shape_raw[1]] = sememe_mat
    mat = np.zeros((lattice_len_max, sense_len_max), dtype=np.int)
    self_mat = np.zeros((sense_len_max, sense_len_max), dtype=np.int)
    if use_default_sense:
        self_mat[0][0] = 1
    for i, tuple_i in enumerate(sense_list):
        if use_default_sense:
            mat[i][0] = 1
        if tuple_i == []:
            continue
        s = tuple_i[0]
        e = tuple_i[1]
        self_mat[s:e, s:e] = 1
        for j in range(s, e):
            mat[i][j] = 1
    return sememe_mat_new, mat, self_mat


def pad_edge_matrix(edge, dim):
    f, b = edge
    assert f.shape[0] == b.shape[0]
    pad_size = dim - f.shape[0]
    assert pad_size >= 0

    padded_f = np.pad(f, ((0, pad_size), (0, pad_size)), 'constant')
    padded_b = np.pad(b, ((0, pad_size), (0, pad_size)), 'constant')
    return [padded_f, padded_b]



