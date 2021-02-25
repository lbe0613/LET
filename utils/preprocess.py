#! -*- coding: utf-8 -*-
import jieba
import thulac
import pkuseg
import getopt
import sys
import json
from tqdm import tqdm
import mxnet as mx
import gluonnlp as nlp

import numpy as np
from gluonnlp.data import BERTSentenceTransform


class BERTDatasetTransform(object):
    """Dataset transformation for BERT-style sentence classification or regression.

    Parameters
    ----------
    tokenizer : BERTTokenizer.
        Tokenizer for the sentences.
    max_seq_length : int.
        Maximum sequence length of the sentences.
    labels : list of int , float or None. defaults None
        List of all label ids for the classification task and regressing task.
        If labels is None, the default task is regression
    pad : bool, default True
        Whether to pad the sentences to maximum length.
    pair : bool, default True
        Whether to transform sentences or sentence pairs.
    label_dtype: int32 or float32, default float32
        label_dtype = int32 for classification task
        label_dtype = float32 for regression task
    """

    def __init__(self,
                 tokenizer,
                 max_seq_length,
                 class_labels=None,
                 label_alias=None,
                 pad=True,
                 pair=True,
                 has_label=True):
        self.class_labels = class_labels
        self.has_label = has_label
        self._label_dtype = 'int32' if class_labels else 'float32'
        if has_label and class_labels:
            self._label_map = {}
            for (i, label) in enumerate(class_labels):
                self._label_map[label] = i
            if label_alias:
                for key in label_alias:
                    self._label_map[key] = self._label_map[label_alias[key]]
        self._bert_xform = BERTSentenceTransform(
            tokenizer, max_seq_length, pad=pad, pair=pair)

    def __call__(self, line):
        """Perform transformation for sequence pairs or single sequences.

        The transformation is processed in the following steps:
        - tokenize the input sequences
        - insert [CLS], [SEP] as necessary
        - generate type ids to indicate whether a token belongs to the first
          sequence or the second sequence.
        - generate valid length

        For sequence pairs, the input is a tuple of 3 strings:
        text_a, text_b and label.

        Inputs:
            text_a: 'is this jacksonville ?'
            text_b: 'no it is not'
            label: '0'
        Tokenization:
            text_a: 'is this jack ##son ##ville ?'
            text_b: 'no it is not .'
        Processed:
            tokens:  '[CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]'
            type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            valid_length: 14
            label: 0

        For single sequences, the input is a tuple of 2 strings: text_a and label.
        Inputs:
            text_a: 'the dog is hairy .'
            label: '1'
        Tokenization:
            text_a: 'the dog is hairy .'
        Processed:
            text_a:  '[CLS] the dog is hairy . [SEP]'
            type_ids: 0     0   0   0  0     0 0
            valid_length: 7
            label: 1

        Parameters
        ----------
        line: tuple of str
            Input strings. For sequence pairs, the input is a tuple of 3 strings:
            (text_a, text_b, label). For single sequences, the input is a tuple
            of 2 strings: (text_a, label).

        Returns
        -------
        np.array: input token ids in 'int32', shape (batch_size, seq_length)
        np.array: valid length in 'int32', shape (batch_size,)
        np.array: input token type ids in 'int32', shape (batch_size, seq_length)
        np.array: classification task: label id in 'int32', shape (batch_size, 1),
            regression task: label in 'float32', shape (batch_size, 1)
        """
        if self.has_label:
            input_ids, valid_length, segment_ids = self._bert_xform(line[:-1])
            label = line[-1]
            # map to int if class labels are available
            if self.class_labels:
                label = self._label_map[label]
            label = np.array([label], dtype=self._label_dtype)
            return input_ids, valid_length, segment_ids, label
        else:
            return self._bert_xform(line)


def get_p_dict(word_seq):
    p_head_dict = {}
    p_tail_dict = {}
    for t in word_seq:
        if not t[1] in p_head_dict:
            p_head_dict[t[1]] = []
        if not t[2] in p_tail_dict:
            p_tail_dict[t[2]] = []
        p_head_dict[t[1]].append(t)
        p_tail_dict[t[2]].append(t)
    return p_head_dict, p_tail_dict


def produce(sen, referlist):
    word_list = []
    ignore = {'[UNK]'}
    start = 0
    start_i = 0
    for i in range(len(sen)):
        if i < start_i:
            continue
        char = sen[i]
        c_len = len(char)
        if char in ignore:
            continue
        for j in range(start, len(referlist)):
            word = referlist[j]
            w_len = len(word)
            flag = 0
            if char == word[0:c_len]:
                for k in range(i + 1, len(sen) + 1):
                    word_t = ''.join(sen[i:k])
                    if word_t == word:
                        word_list.append((word_t, i + 1, k + 1))
                        start_i = k
                        start = j + 1
                        flag = 1
                        break
                    if len(word_t) >= w_len:
                        break
                if flag:
                    break
    return word_list


def cut(input_path, output_path, cut):
    out = open(output_path, 'w')
    p = []

    with open(input_path, 'r') as f:
        thu = thulac.thulac(seg_only=True)
        seg = pkuseg.pkuseg()
        lines = tqdm(f.readlines())
        for l in lines:
            a = {}
            label, sen1, sen2 = l.strip().split('\t')
            sen1 = sen1.strip().replace(' ', '').lower()
            sen2 = sen2.strip().replace(' ', '').lower()
            char_id, valid_len, segment = transform(([sen1, sen2]))
            first_len = (valid_len - segment.sum())
            second_len = segment.sum()
            sentence1_list = tokenizer(sen1)
            sentence1_list = [w.strip("##") for w in sentence1_list]
            sentence1_list = sentence1_list[:first_len - 2]
            sentence2_list = tokenizer(sen2)
            sentence2_list = [w.strip("##") for w in sentence2_list]
            sentence2_list = sentence2_list[:second_len - 1]

            for m in range(2):
                sen = [sentence1_list, sentence2_list][m]
                sentence = [sen1, sen2][m]
                word_list = []
                edge = set()
                if cut == 'jieba':
                    referlist = list(jieba.cut(sentence))
                elif cut == 'thulac':
                    referlist = thu.cut(sentence, text=True).split(' ')
                elif cut == 'pkuseg':
                    referlist = list(seg.cut(sentence))
                elif cut == 'all':
                    referlist1 = list(jieba.cut(sentence))
                    referlist2 = thu.cut(sentence, text=True).split(' ')
                    referlist3 = list(seg.cut(sentence))
                word_list.append(('<s>', 0, 1))
                word_list.append(('</s>', len(sen) + 1, len(sen) + 2))

                if cut == 'all':
                    word_list.extend(produce(sen, referlist1))
                    word_list.extend(produce(sen, referlist2))
                    word_list.extend(produce(sen, referlist3))
                    word_list = list(set(word_list))

                else:
                    word_list.extend(produce(sen, referlist))

                p.append(len(word_list))
                index_set = {(word[1], word[2]): i for i, word in enumerate(word_list)}

                p_head_dict, p_tail_dict = get_p_dict(word_list)

                all_node_keys = set([tt for t in word_list for tt in [t[1], t[2]]])
                all_node_keys = sorted(list(all_node_keys))
                all_node = {idt: {'id': idt} for idt in all_node_keys}
                for k in all_node:
                    t = all_node[k]
                    t['start'] = p_head_dict.get(t['id'], [])
                    t['end'] = p_tail_dict.get(t['id'], [])

                for t in all_node:
                    for tt in all_node[t]['start']:
                        for ttt in all_node[tt[2]]['start']:
                            edge.add((index_set[(tt[1], tt[2])], index_set[(ttt[1], ttt[2])]))

                wordlist = [(word[0], word[1], word[2] - 1) for word in word_list]
                a[('sentence%d' % m)] = {'sentence': sentence, 'word_list': wordlist, 'edge': list(edge)}
            a['label'] = label
            a['first_len'] = str(first_len)
            a['second_len'] = str(second_len + 1)
            a['sent'] = list(char_id.astype(float))
            a['segment'] = list(segment.astype(float))
            out.write(json.dumps(a, ensure_ascii=False) + '\n')

    out.close()


if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('python utils/preprocess.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('python utils/preprocess.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg

    bert, vocab = nlp.model.get_model('bert_12_768_12', dataset_name='wiki_cn_cased',
                                      pretrained=True, ctx=mx.cpu(), use_pooler=True,
                                      use_decoder=False, use_classifier=False)
    tokenizer = nlp.data.BERTTokenizer(vocab, lower=True)
    max_len = 128
    transform = BERTDatasetTransform(tokenizer, max_len, pad=False, pair=True, has_label=False)
    seg_mode = ['all']
    for f in seg_mode:
        input_path = inputfile
        output_path = outputfile

        print('=' * 89)
        print('Start')
        cut(input_path, output_path, f)
