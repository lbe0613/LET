#! -*- coding: utf-8 -*-

from attention_lib import *


class LET(nn.Block):
    """docstring for LET"""

    def __init__(self, sememe_size, layer_size, nb_class, nb_head, nb_layer=None, ctx=None, **kwargs):
        super(LET, self).__init__(**kwargs)

        self.nb_layer = nb_layer if nb_layer is not None else 1
        self.nb_head = nb_head
        self.nb_class = nb_class
        self.layer_size = layer_size
        self.ctx = ctx
        self.bert = None
        with self.name_scope():
            self.sememe_embedding = nn.Embedding(sememe_size, 200)
            self.Dense_sense = nn.Dense(layer_size, flatten=False)
            self.encoder_char = MultiDimEncodingSentence(768, out_dim=layer_size)
            self.transformer_sememe = AttentionBlock(layer_size, nb_head, pooling=True)  # transformer3
            self.transformer_char = BimpmBlock(layer_size, nb_head, pooling=True)  # transformer2
            self.Dense = nn.Dense(layer_size, use_bias=False, flatten=False)

            self.gate_sense = GRU(layer_size)
            self.gate_lattice = GRU(layer_size)
            for i in range(self.nb_layer):
                setattr(self, f'SenseBlock_{i}', eval('FastDiSA')(layer_size, 2, has_mlp=False))
                setattr(self, f'LatticeBlock_{i}', eval('FastDiSA')(layer_size, 1, has_mlp=True))

            self.encoder_lattice = MultiDimEncodingSentence(layer_size, dropout_axes=(0, 2))  # encoder
            self.score = ScoreLayer(layer_size * 2, nb_class=nb_class)

    def init_pretrained_bert(self, bert_model):
        self.bert = bert_model

    def forward(self, feed_dict):
        sent = feed_dict['sent']
        segment = feed_dict['segment']
        valid_len = feed_dict['valid_len']

        sent_len = nd.concat(feed_dict['sent1_len'], feed_dict['sent2_len'], dim=0)
        convert_mat1 = feed_dict['convert_mat1']
        convert_mat2 = feed_dict['convert_mat2']

        lattice = nd.concat(feed_dict['lattice1'], feed_dict['lattice2'], dim=0)
        lattice_len = nd.concat(feed_dict['lattice1_len'], feed_dict['lattice2_len'], dim=0)

        sememe = nd.concat(feed_dict['sememe1'], feed_dict['sememe2'], dim=0)
        sememe_len = nd.concat(feed_dict['sememe1_len'], feed_dict['sememe2_len'], dim=0)
        sememe_to_sense = nd.concat(feed_dict['sense1'], feed_dict['sense2'], dim=0)
        sense_len = nd.concat(feed_dict['sense1_len'], feed_dict['sense2_len'], dim=0)
        sense_map = nd.concat(feed_dict['sense1_map'], feed_dict['sense2_map'], dim=0)

        edge1_b, edge1_f = nd.split(feed_dict['edge1'], axis=1, num_outputs=2)
        edge2_b, edge2_f = nd.split(feed_dict['edge2'], axis=1, num_outputs=2)
        sense1_T = nd.transpose(feed_dict['sense1_map'], (0, 2, 1))
        sense2_T = nd.transpose(feed_dict['sense2_map'], (0, 2, 1))
        edge1_b = nd.expand_dims(nd.batch_dot(sense1_T, nd.squeeze(edge1_b, axis=1)), axis=1)
        edge1_f = nd.expand_dims(nd.batch_dot(sense1_T, nd.squeeze(edge1_f, axis=1)), axis=1)
        edge1 = nd.concat(edge1_b, edge1_f, dim=1)
        edge2_b = nd.expand_dims(nd.batch_dot(sense2_T, nd.squeeze(edge2_b, axis=1)), axis=1)
        edge2_f = nd.expand_dims(nd.batch_dot(sense2_T, nd.squeeze(edge2_f, axis=1)), axis=1)
        edge2 = nd.concat(edge2_b, edge2_f, dim=1)
        edge = nd.concat(edge1, edge2, dim=0)
        lattice_to_char = nd.transpose(lattice, (0, 2, 1))

        # Char Sequence
        char_emb, sent_emb = self.bert(sent, segment, valid_len)
        char_emb1 = nd.batch_dot(convert_mat1, char_emb)
        char_emb2 = nd.batch_dot(convert_mat2, char_emb)
        char_emb = nd.concat(char_emb1, char_emb2, dim=0)

        shape_raw = char_emb.shape
        char_shape_raw = (2, int(shape_raw[0] / 2), shape_raw[1], shape_raw[2])

        # char_emb: b * sentence_len * dim, H: b * lattice_len * dim
        _, word_emb = self.encoder_char(char_emb, sent_len, char_shape_raw, lattice)

        # sememe
        sememe_emb = self.sememe_embedding(sememe)
        sememe_emb = self.Dense_sense(sememe_emb)
        shape_raw = sememe_emb.shape
        sememe_shape_raw = (2, int(shape_raw[0] / 2), shape_raw[1], shape_raw[2])
        Y_sememe, sen_emb, H_sememe = self.transformer_sememe(sememe_emb, sememe_len, sememe_shape_raw, sememe_to_sense)

        # Lattice
        shape_raw = word_emb.shape
        word_shape_raw = (2, int(shape_raw[0] / 2), shape_raw[1], shape_raw[2])
        shape_raw = sen_emb.shape
        sense_shape_raw = (2, int(shape_raw[0] / 2), shape_raw[1], shape_raw[2])

        for i in range(self.nb_layer):
            # update sense
            H, A_sense = getattr(self, f'SenseBlock_{i}')(sen_emb, word_emb, word_emb, sense_len, lattice_len,
                                                          sense_shape_raw, edge)
            sen_emb = self.gate_sense(H, sen_emb, sense_len)

            # update lattice
            H, A_lattice = getattr(self, f'LatticeBlock_{i}')(word_emb, sen_emb, sen_emb, lattice_len, sense_len,
                                                              word_shape_raw,
                                                              nd.expand_dims(sense_map, axis=1))
            word_emb = self.gate_lattice(H, word_emb, lattice_len)

        S, H = self.encoder_lattice(word_emb, lattice_len, word_shape_raw, lattice_to_char)
        H = H + self.Dense(char_emb)

        Y_char = self.transformer_char(H, sent_len, char_shape_raw)

        Y2 = nd.split(Y_char, axis=0, num_outputs=2)
        P1 = sent_emb
        P2 = nd.concat(Y2[0], Y2[1], nd.abs(Y2[0] - Y2[1]), Y2[0] * Y2[1], dim=-1)
        prob = self.score(nd.concat(P1, P2, dim=-1))

        return prob
