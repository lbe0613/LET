#! -*- coding: utf-8 -*-

import os
import getopt
import random
import configparser
import time
import logging
import json
import sys

sys.path.append('./model')

import mxnet as mx
from mxnet import autograd
import mxnet.gluon as gl
import gluonnlp as nlp
from model.LET_model import *
from utils.gen_matrix import *


def load_bert(ctx):
    bert, vocab = nlp.model.get_model('bert_12_768_12', dataset_name='wiki_cn_cased',
                                      pretrained=True, ctx=ctx, use_pooler=True,
                                      use_decoder=False, use_classifier=False)
    tokenizer = nlp.data.BERTTokenizer(vocab, lower=True)
    return bert, tokenizer, vocab


def sense2index(lattice, sense_dict, sememe_dict):
    sense_ids = [0]
    sense_list = []
    sememe_ids = [0]
    for word in lattice:
        if word in sense_dict:
            id_list = []
            for x, y in sense_dict[word].items():
                id_list.append(int(x))
                sememe_ids.extend(y)
            sense_ids.extend(id_list)
            sense_list.append([len(sense_ids) - len(sense_dict[word]), len(sense_ids)])

        else:
            sense_list.append([])

    sememe_ids = list(set(sememe_ids))
    sememe_matrix = np.zeros(shape=(len(sense_ids), len(sememe_ids)))
    for i in range(len(sense_ids)):
        id = str(sense_ids[i])
        if id == '0':
            continue
        for sememe in sememe_dict[id]:
            j = sememe_ids.index(sememe)
            sememe_matrix[i][j] = 1

    return sense_ids, sense_list, sememe_ids, sememe_matrix


def gen_data_set(data_dir, sense_dict, sememe_dict, EDGE_MODE='reachable'):
    data_file = open(data_dir, 'r')

    data_set = {
        'sent': [],
        'segment': [],
        'valid_len': [],
        'sent1_len': [],
        'sent2_len': [],
        'lattice1': [],
        'lattice2': [],
        'lattice1_len': [],
        'lattice2_len': [],
        'edge1': [],
        'edge2': [],
        'label': [],
        'origin_1': [],
        'origin_2': [],
        'sense1_id': [],
        'sense2_id': [],
        'sense1_len': [],
        'sense2_len': [],
        'sense1_list': [],
        'sense2_list': [],
        'sememe1_id': [],
        'sememe2_id': [],
        'sememe1_len': [],
        'sememe2_len': [],
        'sememe1_mat': [],
        'sememe2_mat': []
    }

    for l in data_file.readlines():
        line = json.loads(l)
        data_set['label'].append(int(line['label']))
        first_len = int(line['first_len'])
        second_len = int(line['second_len'])
        data_set['sent1_len'].append(first_len)
        data_set['sent2_len'].append(second_len)

        word1_list = line['sentence0']['word_list']
        word2_list = line['sentence1']['word_list']

        edge1 = line['sentence0']['edge']
        edge2 = line['sentence1']['edge']

        ad_edge1, edge1 = gen_edge_matrix(edge1, len(word1_list), edge_mode=EDGE_MODE)
        ad_edge2, edge2 = gen_edge_matrix(edge2, len(word2_list), edge_mode=EDGE_MODE)

        char_id = np.array(line['sent'], dtype='int')
        segment = np.array(line['segment'], dtype='int')

        assert second_len == segment.sum() + 1

        data_set['sent'].append(char_id)
        data_set['segment'].append(segment)
        data_set['valid_len'].append(len(char_id))

        data_set['edge1'].append(edge1)
        data_set['edge2'].append(edge2)

        data_set['lattice1'].append(word1_list)
        data_set['lattice2'].append(word2_list)

        lattice1_list = [t[0].strip() for t in word1_list]
        lattice2_list = [t[0].strip() for t in word2_list]

        data_set['lattice1_len'].append(len(word1_list))
        data_set['lattice2_len'].append(len(word2_list))

        sense1_id, sense1_list, sememe1_id, sememe1_mat = sense2index(lattice1_list, sense_dict, sememe_dict)
        sense2_id, sense2_list, sememe2_id, sememe2_mat = sense2index(lattice2_list, sense_dict, sememe_dict)

        data_set['sense1_id'].append(sense1_id)
        data_set['sense2_id'].append(sense2_id)
        data_set['sense1_len'].append(len(sense1_id))
        data_set['sense2_len'].append(len(sense2_id))
        data_set['sense1_list'].append(sense1_list)
        data_set['sense2_list'].append(sense2_list)
        data_set['sememe1_id'].append(sememe1_id)
        data_set['sememe2_id'].append(sememe2_id)
        data_set['sememe1_len'].append(len(sememe1_id))
        data_set['sememe2_len'].append(len(sememe2_id))
        data_set['sememe1_mat'].append(sememe1_mat)
        data_set['sememe2_mat'].append(sememe2_mat)
    data_file.close()

    print(f'loading data {data_dir} is complete!')

    return data_set


def load_data(train, dev, test, sense_dict, edge_mode=None):
    sememe_dict = {}
    for word in sense_dict:
        for key, value in sense_dict[word].items():
            sememe_dict[key] = value

    data_sets = {}
    data_sets['train'] = gen_data_set(train, sense_dict, sememe_dict, EDGE_MODE=edge_mode)
    data_sets['valid'] = gen_data_set(dev, sense_dict, sememe_dict, EDGE_MODE=edge_mode)
    data_sets['test'] = gen_data_set(test, sense_dict, sememe_dict, EDGE_MODE=edge_mode)

    return data_sets, sememe_dict


def load_sememe_embedding(sememe_embedding_file, embedding_size, ctx=None):
    with open(sememe_embedding_file, 'r') as file:
        f = file.readlines()[1:]
    matrix = np.random.randn(len(f) + 1, embedding_size) * 0.01

    for i, line in enumerate(f):
        line = line.strip().split(' ')[1:]
        assert len(line) == embedding_size

        vector = [float(num.strip()) for num in line]
        vector = np.array(vector)

        matrix[i + 1] = vector

    return nd.array(matrix, ctx=ctx)


def data_iter(data_size, batch_size, shuffle=True):
    idx = list(range(data_size))
    if shuffle is True:
        random.shuffle(idx)
    for i in range(0, data_size, batch_size):
        j = idx[i:min(i + batch_size, data_size)]
        yield j


def gen_batch(data, id_batch, use_default_sense, ctx):
    batch = {
        'sent': [],
        'segment': [],
        'valid_len': [],
        'sent1_len': [],
        'sent2_len': [],
        'convert_mat1': [],
        'convert_mat2': [],
        'lattice1': [],
        'lattice2': [],
        'sememe1': [],
        'sememe2': [],
        'sememe1_len': [],
        'sememe2_len': [],
        'edge1': [],
        'edge2': [],
        'lattice1_len': [],
        'lattice2_len': [],
        'sense1': [],
        'sense2': [],
        'sense1_len': [],
        'sense2_len': [],
        'sense1_map': [],
        'sense2_map': [],
        'pos1_s': [],
        'pos1_e': [],
        'pos2_s': [],
        'pos2_e': []
    }

    sent_len = [data['valid_len'][i] for i in id_batch]
    sent_len_max = max(sent_len)

    sent1_len = [data['sent1_len'][i] for i in id_batch]
    sent2_len = [data['sent2_len'][i] for i in id_batch]
    single_sent_len_max = max(sent1_len + sent2_len)

    lattice1_len = [data['lattice1_len'][i] for i in id_batch]
    lattice2_len = [data['lattice2_len'][i] for i in id_batch]
    lattice_len_max = max(lattice1_len + lattice2_len)

    sense1_len = [data['sense1_len'][i] for i in id_batch]
    sense2_len = [data['sense2_len'][i] for i in id_batch]
    sense_len_max = max(sense1_len + sense2_len)

    sememe1_len = [data['sememe1_len'][i] for i in id_batch]
    sememe2_len = [data['sememe2_len'][i] for i in id_batch]
    sememe_len_max = max(sememe1_len + sememe2_len)

    for i in id_batch:
        sememe1 = [0] * sememe_len_max
        sememe1[:data['sememe1_len'][i]] = data['sememe1_id'][i]

        sememe2 = [0] * sememe_len_max
        sememe2[:data['sememe2_len'][i]] = data['sememe2_id'][i]

        sense1 = [0] * sense_len_max
        sense1[:data['sense1_len'][i]] = data['sense1_id'][i]

        sense2 = [0] * sense_len_max
        sense2[:data['sense2_len'][i]] = data['sense2_id'][i]

        sent = [0] * sent_len_max
        sent[:data['valid_len'][i]] = data['sent'][i]

        segment = [0] * sent_len_max
        segment[:data['valid_len'][i]] = data['segment'][i]

        lattice1, forward_position1, backward_position1, pos1_s, pos1_e = gen_lattice_map(data['lattice1'][i],
                                                                                          lattice_len_max,
                                                                                          single_sent_len_max)
        lattice2, forward_position2, backward_position2, pos2_s, pos2_e = gen_lattice_map(data['lattice2'][i],
                                                                                          lattice_len_max,
                                                                                          single_sent_len_max)

        sense1, sense1_map, self1_map = gen_sense_map(data['sememe1_mat'][i], data['sense1_list'][i], lattice_len_max,
                                                      sense_len_max, sememe_len_max, use_default_sense)
        sense2, sense2_map, self2_map = gen_sense_map(data['sememe2_mat'][i], data['sense2_list'][i], lattice_len_max,
                                                      sense_len_max, sememe_len_max, use_default_sense)

        edge1 = pad_edge_matrix(data['edge1'][i], lattice_len_max)
        edge2 = pad_edge_matrix(data['edge2'][i], lattice_len_max)
        convert_mat1, convert_mat2 = gen_convert_map(data['sent1_len'][i], data['sent2_len'][i], single_sent_len_max,
                                                     sent_len_max)

        batch['sent'].append(sent)
        batch['sememe1'].append(sememe1)
        batch['sememe2'].append(sememe2)
        batch['sememe1_len'].append(data['sememe1_len'][i])
        batch['sememe2_len'].append(data['sememe2_len'][i])

        batch['edge1'].append(edge1)
        batch['edge2'].append(edge2)
        batch['valid_len'].append(data['valid_len'][i])
        batch['segment'].append(segment)

        batch['lattice1'].append(lattice1)
        batch['lattice2'].append(lattice2)
        batch['lattice1_len'].append(data['lattice1_len'][i])
        batch['lattice2_len'].append(data['lattice2_len'][i])
        batch['sense1'].append(sense1)
        batch['sense2'].append(sense2)
        batch['sense1_len'].append(data['sense1_len'][i])
        batch['sense2_len'].append(data['sense2_len'][i])
        batch['sense1_map'].append(sense1_map)
        batch['sense2_map'].append(sense2_map)
        batch['sent1_len'].append(data['sent1_len'][i])
        batch['sent2_len'].append(data['sent2_len'][i])
        batch['convert_mat1'].append(convert_mat1)
        batch['convert_mat2'].append(convert_mat2)
        batch['pos1_s'].append(pos1_s)
        batch['pos2_s'].append(pos2_s)
        batch['pos1_e'].append(pos1_e)
        batch['pos2_e'].append(pos2_e)

    for key in batch:
        batch[key] = nd.array(batch[key], ctx=ctx)

    return batch


def evaluate_data_set(net, data_set, ctx=mx.gpu(0)):
    data_size = len(data_set['label'])
    result = [0, 0, 0, 0]
    ce_loss = 0.0
    for id_batch in data_iter(data_size, 100, shuffle=False):
        feed_dict = gen_batch(data_set, id_batch, use_default_sense, ctx)
        label = nd.array([data_set['label'][i] for i in id_batch], ctx=ctx)

        with autograd.predict_mode():
            predict_prob = net(feed_dict)

        l = loss_function(predict_prob, label).sum()
        ce_loss += l.asscalar()

        if nb_class == 2:
            predict = nd.argmax(predict_prob, axis=-1)
            Y = predict + label * 2
            for i in range(len(id_batch)):
                result[Y[i].asscalar().astype(int)] += 1

    ce_loss /= data_size
    return result, ce_loss


def format_result(result, loss):
    data_size = sum(result)
    acc = 1.0 * (result[0] + result[3]) / data_size
    R = 1.0 * result[3] / (result[2] + result[3] + 1e-16)
    P = 1.0 * result[3] / (result[1] + result[3] + 1e-16)
    F = 2.0 * R * P / (R + P + 1e-16)

    return acc, F, f'loss:{loss:.4f}, {result[0]}/{result[1]}/{result[2]}/{result[3]}, ACC:{acc:.4f}, P/R/F:{P:.4f}/{R:.4f}/{F:.4f}'


if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "-h-c:-d:")
    except getopt.GetoptError:
        print('python3.6 train_sememe.py -c <configfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('python3.6 train_sememe.py -c <configfile>')
            sys.exit()
        elif opt in ("-c", "--config"):
            conf_file = arg

    cf = configparser.ConfigParser()
    cf.read(conf_file)

    seed = cf.getint('network_hypeparameter', 'seed')

    mx.random.seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))

    ct = cf.get('train', 'ctx')
    if ct == 'cpu':
        ctx = mx.cpu()
    else:
        gpu_id = cf.getint('train', 'gpu_id')
        if gpu_id >= 0:
            ctx = mx.gpu(gpu_id)
        else:
            ctx = mx.gpu()

    logging.getLogger().setLevel(logging.INFO)
    train_dir = cf.get('data', 'train')
    dev_dir = cf.get('data', 'dev')
    test_dir = cf.get('data', 'test')

    sememe_embedding_file = cf.get('network_hypeparameter', 'sememe_embedding')
    sense_dict_file = cf.get('network_hypeparameter', 'sense_dict')
    with open(sense_dict_file, 'r') as f:
        sense_dict = json.load(f)

    sememe_embedding_matrix = load_sememe_embedding(sememe_embedding_file, 200, ctx=ctx)
    sememe_size = sememe_embedding_matrix.shape[0]
    print(sememe_size)

    bert_model, tokenizer, vocab = load_bert(ctx)

    EDGE_MODE = cf.get('network_hypeparameter', 'edge_mode')
    data_sets, sememe_dict = load_data(train_dir, dev_dir, test_dir, sense_dict, edge_mode=EDGE_MODE)

    train_data = data_sets['train']
    train_size = len(train_data['label'])

    nb_head = cf.getint('network_hypeparameter', 'nb_head')
    nb_layer = cf.getint('network_hypeparameter', 'nb_layer')
    use_default_sense = cf.getboolean('network_hypeparameter', 'use_default_sense')
    layer_size = cf.getint('network_hypeparameter', 'layer_size')
    nb_class = cf.getint('network_hypeparameter', 'nb_class')

    net = LET(sememe_size, layer_size, nb_class, nb_head, nb_layer, ctx)
    net.collect_params().initialize(ctx=ctx)
    net.init_pretrained_bert(bert_model)
    net.collect_params()['let0_embedding0_weight'].set_data(sememe_embedding_matrix)
    loss_function = gl.loss.SoftmaxCELoss()

    bert_lr_mult = cf.getfloat('train', 'bert_lr_mult')
    for param in net.bert.collect_params().values():
        param.lr_mult = bert_lr_mult

    all_model_params = net.collect_params()

    epochs = cf.getint('train', 'epochs')
    batch_size = cf.getint('train', 'batch_size')
    learning_rate = cf.getfloat('train', 'learning_rate')
    weight_decay = cf.getfloat('train', 'weight_decay')
    optimizer = cf.get('train', 'optimizer', fallback='adam')

    epsilon = 1e-6
    log_interval = cf.getint('train', 'log_interval')
    accumulate = cf.getint('train', 'accumulate')

    num_train_examples = train_size
    warmup_ratio = cf.getfloat('train', 'warmup_ratio')
    step_size = batch_size * accumulate if accumulate else batch_size
    num_train_steps = int(num_train_examples / step_size * epochs)
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    step_num = 0

    trainer = gl.Trainer(all_model_params, optimizer,
                         {'learning_rate': learning_rate, 'epsilon': epsilon, 'wd': weight_decay})
    for _, v in net.bert.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    params = [p for p in all_model_params.values() if p.grad_req != 'null']

    if accumulate and accumulate > 1:
        for p in params:
            p.grad_req = 'add'

    log_file_dir = cf.get('log', 'dir')
    if not os.path.exists(log_file_dir):
        os.makedirs(log_file_dir)
    localtime = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    head, tail = os.path.split(conf_file)
    log_file = os.path.join(log_file_dir, tail.strip('.conf') + '_' + localtime + str(seed) + '.log')
    f_log = open(log_file, 'w')

    model_root = cf.get('model', 'dir')
    model_dir = os.path.join(model_root, tail.strip('.conf') + '/' + localtime + str(seed))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    logging.info(
        f'sememe model epoch={epochs},lr={learning_rate},optimizer={optimizer},warm_ratio={warmup_ratio},bert_lr_mult={bert_lr_mult},log_dir={log_file_dir}')

    valid_result = {'F': [], 'acc': []}

    for epoch in range(epochs):
        step_loss = 0
        # result,loss = evaluate_data_set(net, data_sets['train'], ctx=ctx)
        # train_acc, train_f, train_result_str = format_result(result, loss)

        for batch_id, id_batch in enumerate(data_iter(train_size, batch_size)):

            if step_num < num_warmup_steps:
                new_lr = learning_rate * step_num / num_warmup_steps
            else:
                non_warmup_steps = step_num - num_warmup_steps
                offset = non_warmup_steps / (num_train_steps - num_warmup_steps)
                new_lr = learning_rate - offset * learning_rate
            trainer.set_learning_rate(new_lr)

            feed_dict = gen_batch(train_data, id_batch, use_default_sense, ctx)

            label = nd.array([train_data['label'][i] for i in id_batch], ctx=ctx)

            with autograd.record():
                with autograd.train_mode():
                    predict = net(feed_dict)
                loss = loss_function(predict, label).mean()
                loss.backward()

            if not accumulate or (batch_id + 1) % accumulate == 0:
                trainer.step(accumulate if accumulate else 1)
                step_num += 1
                if accumulate and accumulate > 1:
                    # set grad to zero for gradient accumulation
                    all_model_params.zero_grad()
            step_loss += loss.asscalar()

            if (batch_id + 1) % (log_interval) == 0:
                step_loss = 0

                result, loss = evaluate_data_set(net, data_sets['valid'], ctx=ctx)
                valid_acc, valid_f, valid_result_str = format_result(result, loss)

                if epoch > 1:
                    if valid_acc > max(valid_result['acc']):
                        net.save_parameters(os.path.join(model_dir, f'best_valid_acc_{epoch + 1}_{step_num}.params'))

                valid_result['acc'].append(valid_acc)

                result, loss = evaluate_data_set(net, data_sets['test'], ctx=ctx)

                test_acc, test_f, test_result_str = format_result(result, loss)
                f_log.write(
                    f'epoch {epoch + 1}: id {step_num}:\tVALID: {valid_result_str}\tTEST: {test_result_str}\n\n')
                f_log.flush()
                print(f'epoch {epoch + 1}: id {step_num}:\tVALID: {valid_result_str}\tTEST: {test_result_str}\n\n')

        mx.nd.waitall()

    f_log.close()
