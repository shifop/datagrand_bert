# coding: utf-8
"""bilstm-crf 用于词性标注"""

from datetime import timedelta
from ner.tools import *
import numpy as np
import tensorflow as tf
import time
import math
from tqdm import tqdm
import json
import tensorflow.contrib.keras as kr
import os
from bert.modeling import *


def parser(record):
    features = tf.parse_single_example(record,
                                       features={
                                           'seq': tf.FixedLenFeature([202], tf.int64),
                                           'ft': tf.FixedLenFeature([], tf.string),
                                           'tag': tf.FixedLenFeature([202], tf.int64),
                                           'mask': tf.FixedLenFeature([200], tf.int64)
                                       }
                                       )
    return features['seq'], features['ft'], features['tag'], features['mask']


def parser_dev(record):
    features = tf.parse_single_example(record,
                                       features={
                                           'seq': tf.FixedLenFeature([202], tf.int64),
                                           'ft': tf.FixedLenFeature([], tf.string)
                                       }
                                       )
    return features['seq'], features['ft']


def norm(x, scope, *, axis=-1, epsilon=1e-5):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        n_state = x.shape[-1].value
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x - u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x * g + b
        return x


class TextCNN(object):

    def __init__(self, config, split_fn):
        self.config = config
        self.split = split_fn
        self.__createModel()
        self.log_writer = tf.summary.FileWriter('./log/20190511214400', self.graph)
        self.train_data = {}
        self.test_data = {}

    def initialize(self, save_path):
        with self.graph.as_default():
            saver = tf.train.Saver(name='save_saver')
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            saver.save(sess, save_path)

    def _dcnn(self, seq, kernel, rate, name, training, dropout=True):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            alpha = tf.layers.conv1d(seq, self.config.embedding_size*2+9, kernel, activation=tf.nn.sigmoid, dilation_rate=rate,
                                     padding='SAME', name = 'alpha', reuse= tf.AUTO_REUSE)
            seq_cnn = tf.layers.conv1d(seq, self.config.embedding_size*2+9, kernel, dilation_rate=rate,
                                     padding='SAME', name = 'cnn_encode', reuse= tf.AUTO_REUSE)
            output = seq*(1-alpha)+seq_cnn*alpha
            output = norm(output, 'norm')
            if dropout:
                output = tf.layers.dropout(output, 0.1, training=training)
            # output = tf.nn.relu(output)
            return output

    def __encode(self, cells, seq, ft, config, training=True):
        """
        使用bilstm编码
        :param seq: 待标注序列
        :param config: 相关配置信息
        :return: h(y_i|X)
        """
        # load bert embedding
        bert_config = BertConfig.from_json_file("./data/config/config.json")  # 配置文件地址。
        model = BertModel(
            config=bert_config,
            is_training=training,
            input_ids=seq,
            use_one_hot_embeddings=False,
            scope='bert')
        embedding = model.get_sequence_output()

        embedding = tf.concat([embedding, ft], axis=-1)

        noise = [tf.shape(embedding)[0], 1, tf.shape(embedding)[2]]
        embedding = tf.layers.dropout(embedding, 0.5, noise, training=training)
        # embedding = norm(embedding, 'em-norm')

        output = self._dcnn(embedding, 5, 1, 'dcnn1', training)
        output = self._dcnn(output, 5, 2, 'dcnn2', training)
        output = self._dcnn(output, 5, 5, 'dcnn3', training)
        output = self._dcnn(output, 5, 1, 'dcnn4', training, False)
        output = tf.tanh(output)
        # output = tf.concat([output,ft], axis=-1)
        h_v = tf.layers.dense(output, config.pos_size, name='dense', reuse=tf.AUTO_REUSE)[:,1:-1,:]
        return h_v

    def __crf_loss(self, seq, tag, tag_p2p, h_v, mask, config):
        """
        计算crf版的loss
        :param seq: 待标注序列
        :param tag: 词性标注 ,shape为[batch, seq_length-1]
        :param h_v: 词-词性得分矩阵
        :param config: 相关配置信息
        :return:
        """
        with tf.name_scope("decode"):
            with tf.variable_scope("var-decode", reuse=tf.AUTO_REUSE):
                # tag-tag 得分矩阵
                mask = tf.argmax(mask, axis=-1)
                seq_mark = tf.sequence_mask(mask, config.seq_length, dtype=tf.float32)
                g_v = tf.get_variable('g_v', [config.pos_size * config.pos_size, 1], tf.float32)
                g_v = tf.reshape(g_v, [config.pos_size, config.pos_size])
                loss, _ = tf.contrib.crf.crf_log_likelihood(h_v, tag, mask, g_v)

        return tf.reduce_mean(-loss)

    def __p(self, h_v, g_v, mask):
        """
        使用动态规划找到最大可能性的标注
        :param h_v: 词-词性得分矩阵 [batch, seq_length, pos_size]
        :param g_v: 词性-词性转移得分矩阵 [pos_size, pos_size]
        :param mask: 有效长度
        :return:
        """
        rt = []
        for index, h in enumerate(h_v):
            pos_size = g_v.shape[0]
            seq_length = mask[index]
            path = [[] for x in range(pos_size)]  # 在当前阶段，标注为不同词性的最大得分的序列
            score = np.zeros([seq_length, pos_size], np.float32)  # 在当前阶段，标注为对应词性的最大得分

            score[0, :] += h[0, :]
            for i in range(pos_size):
                path[i].append(i)
            for index in range(1, seq_length):
                # 计算在当前阶段，标注为不同词性的最大得分
                path_cache = path
                path = [[] for x in range(pos_size)]
                for i in range(pos_size):
                    cache = np.array([score[index - 1, y] + g_v[y, i] + h[index, i] for y in range(pos_size)])
                    max_v = cache.max(axis=0)
                    max_index = cache.argmax(axis=0)
                    # 更新得分
                    score[index, i] = max_v
                    # 更新路径
                    path[i].extend(path_cache[max_index])
                    path[i].append(i)

            max_index = score[seq_length - 1, :].argmax(axis=-1)
            rt.append(path[max_index])
        return rt

    def __get_data(self, path, parser, is_train=False):
        dataset = tf.data.TFRecordDataset(path)
        dataset = dataset.map(parser, num_parallel_calls=4)
        dataset = dataset.batch(self.config.batch_size)
        if is_train:
            dataset = dataset.shuffle(256 * 10)
            dataset = dataset.prefetch(256 * 10)
        iter = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        seq, ft, tag, mask = iter.get_next()

        seq = tf.cast(seq, tf.int32)
        ft = tf.decode_raw(ft, tf.float32)
        tag = tf.cast(tag, tf.int32)
        mask = tf.cast(mask, tf.float32)

        seq = tf.reshape(seq, [-1, self.config.seq_length])
        ft = tf.reshape(ft, [-1, self.config.seq_length, 9])
        tag = tf.reshape(tag, [-1, self.config.seq_length])
        mask = tf.reshape(mask, [-1, self.config.seq_length - 2])

        # mask2 = tf.

        return seq, ft, tag[:, :-2], None, mask, iter.make_initializer(dataset)

    def __get_dev_data(self, path, parser):
        dataset = tf.data.TFRecordDataset(path)
        dataset = dataset.map(parser, num_parallel_calls=4)
        iter = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        seq, ft = iter.get_next()
        # 担心数据不一致，所以转化一次
        seq = tf.reshape(seq, [-1, self.config.seq_length])
        ft = tf.decode_raw(ft, tf.float32)

        ft = tf.reshape(ft, [-1, self.config.seq_length, self.config.pos_size - 1])
        return seq, ft, iter.make_initializer(dataset)

    def __p2(self, cell, seq, ft):
        h_v = self.__encode(cell, seq, ft, self.config, False)
        with tf.name_scope("decode"):
            with tf.variable_scope("var-decode", reuse=tf.AUTO_REUSE):
                # tag-tag 得分矩阵
                g_v = tf.get_variable('g_v', [self.config.pos_size * self.config.pos_size, 1], tf.float32)
                g_v = tf.reshape(g_v, [self.config.pos_size, self.config.pos_size])
        return h_v, g_v

    def __train(self, cell, seq, ft, tag, tag_p2p, mask, training=True):
        h_v = self.__encode(cell, seq, ft, self.config, training)
        loss = self.__crf_loss(seq, tag, tag_p2p, h_v, mask, self.config)
        with tf.name_scope("decode"):
            with tf.variable_scope("var-decode", reuse=tf.AUTO_REUSE):
                # tag-tag 得分矩阵
                g_v = tf.get_variable('g_v', [self.config.pos_size * self.config.pos_size, 1], tf.float32)
                g_v = tf.reshape(g_v, [self.config.pos_size, self.config.pos_size])
        return loss, h_v, g_v

    def __dev(self, inputX, input_index, inputY=None, is_test=False):
        input = tf.split(inputX, 12)
        outputs = self.encode_dev(input[-1], input_index, self.config)

        if is_test:
            return outputs

        else:
            with tf.name_scope("loss"):
                loss = tf.sqrt(tf.reduce_mean(tf.pow(inputY - outputs, 2)))
            return loss

    def __get_train_op(self, var_list1, var_list2, loss):
        global_step = tf.train.get_or_create_global_step()
        learning_rate1 = tf.constant(value=self.config.learning_rate[0], shape=[], dtype=tf.float32)

        # Implements linear decay of the learning rate.
        learning_rate1 = tf.train.polynomial_decay(
            learning_rate1,
            global_step,
            10000,
            end_learning_rate=self.config.learning_rate[0] * 0.0001,
            power=1.0,
            cycle=False)

        learning_rate2 = tf.constant(value=self.config.learning_rate[1], shape=[], dtype=tf.float32)

        # Implements linear decay of the learning rate.
        learning_rate2 = tf.train.polynomial_decay(
            learning_rate2,
            global_step,
            10000,
            end_learning_rate=self.config.learning_rate[1] * 0.001,
            power=1.0,
            cycle=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate1)
        optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate2)
        grads = tf.gradients(loss, var_list1 + var_list2)
        grads1 = grads[:len(var_list1)]
        grads2 = grads[len(var_list1):]
        train_op1 = optimizer.apply_gradients(zip(grads1, var_list1))
        train_op2 = optimizer1.apply_gradients(zip(grads2, var_list2), global_step=global_step)
        train_op = tf.group(train_op1, train_op2)
        return train_op

    def __createModel(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            seq, ft, tag, tag_p2p, self.mask, self.train_data_op = self.__get_data(self.config.train_data_path, parser,
                                                                                   is_train=True)
            dev_seq, dev_ft, dev_tag, dev_tag_p2p, self.dev_mask, self.dev_data_op = self.__get_data(
                self.config.test_data_path, parser)
            test_seq, test_ft, self.p_data_op = self.__get_dev_data(self.config.dev_data_path, parser_dev)

            cells = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(self.config.embedding_size,
                                                                            return_sequences=True))

            # cells = ONLSTM(self.config.embedding_size, 20, True, 0.2)
            # cells = [ONLSTM(self.config.embedding_size, 20, True, 0.2,name="on-lstm-r"),
            #          ONLSTM(self.config.embedding_size, 20, True, 0.2,name="on-lstm-l")]

            self.loss, self.h_v, self.g_v = self.__train(cells, seq, ft, tag, tag_p2p, self.mask, training=True)
            self.dev_loss, self.dev_h_v, self.dev_g_v = self.__train(cells, dev_seq, dev_ft, dev_tag, dev_tag_p2p,
                                                                     self.dev_mask, training=False)

            self.summary_train_loss = tf.summary.scalar('train_loss', self.loss)
            self.summary_dev_loss = tf.summary.scalar('dev_loss', self.dev_loss)

            # 创建用于预测的相关图
            self.seq_placeholder = tf.placeholder(shape=[1, None], dtype=tf.int32, name="seq_placeholder")
            self.ft_placeholder = tf.placeholder(shape=[1, None, 9], dtype=tf.float32, name="ft_placeholder")
            self.split_size = tf.placeholder(shape=(1), dtype=tf.int32, name="split_size")

            self.p_h_v, self.p_g_v = self.__p2(cells, self.seq_placeholder, self.ft_placeholder)

            self.dev_tag = dev_tag

            tv = tf.trainable_variables()

            # 优化器
            self.optim = self.__get_train_op(tv[:103], tv[103:], self.loss)
            # global_step = tf.train.get_or_create_global_step()
            # learning_rate1 = tf.constant(value=self.config.learning_rate[1], shape=[], dtype=tf.float32)
            #
            # # Implements linear decay of the learning rate.
            # learning_rate1 = tf.train.polynomial_decay(
            #     learning_rate1,
            #     global_step,
            #     10000,
            #     end_learning_rate=self.config.learning_rate[1] * 0.01,
            #     power=1.0,
            #     cycle=False)
            # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate1)
            # self.optim = optimizer.minimize(self.loss,var_list=tv[103:], global_step=global_step)

            # self.saver = tf.train.Saver(tf.trainable_variables())
            self.saver = tf.train.Saver(tf.trainable_variables())
            for index, x in enumerate(tf.trainable_variables()):
                print('%d:%s' % (index, x))
            self.saver_v = tf.train.Saver(tf.trainable_variables()[:103])

            self.merged = tf.summary.merge_all()

    def train(self, load_path, save_path, is_reload=False):
        print('Training and evaluating...')
        start_time = time.time()
        total_batch = 0  # 总批次
        min_loss = -1
        last_improved = 0  # 记录上一次提升批次
        test_data_count = get_data_count(self.config.test_data_path)
        train_data_count = get_data_count(self.config.train_data_path)
        require_improvement = 3500  # 如果超过指定轮未提升，提前结束训练
        all_loss = 0

        flag = False
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

            if is_reload:
                sess.run(tf.global_variables_initializer())
                self.saver.restore(sess, load_path)
            else:
                sess.run(tf.global_variables_initializer())
                self.saver_v.restore(sess, load_path)

            for epoch in range(self.config.num_epochs):
                if flag:
                    break
                print('Epoch:', epoch + 1)
                sess.run(self.train_data_op)
                for step in tqdm(range(train_data_count // self.config.batch_size)):
                    if total_batch % self.config.print_per_batch == 0:
                        if total_batch % self.config.dev_per_batch == 0:
                            # 跑验证集
                            dev_loss, dev_acc = self.evaluate(sess, total_batch // self.config.dev_per_batch - 1,
                                                              test_data_count)
                            if min_loss == -1 or min_loss <= dev_acc:
                                self.saver.save(sess=sess, save_path=save_path)
                                improved_str = '*'
                                last_improved = total_batch
                                min_loss = dev_acc
                            else:
                                improved_str = ''

                            time_dif = self.get_time_dif(start_time)
                            msg = 'Iter: {0:>6}, Train Loss: {1:>6.5}, Val loss: {2:>6.5}, Val acc:{3:>6.5} Time: {4} {5}'
                            print(msg.format(total_batch, all_loss / self.config.print_per_batch, dev_loss, dev_acc,
                                             time_dif, improved_str))
                        else:
                            time_dif = self.get_time_dif(start_time)
                            msg = 'Iter: {0:>6}, Train Loss: {1:>6.5}, Time: {2}'
                            print(msg.format(total_batch, all_loss / self.config.print_per_batch, time_dif))
                        all_loss = 0

                    loss_train, summary, _ = sess.run([self.loss, self.summary_train_loss, self.optim])  # 运行优化
                    self.log_writer.add_summary(summary, total_batch)
                    all_loss += loss_train
                    total_batch += 1

                    if total_batch - last_improved > require_improvement:
                        # 验证集正确率长期不提升，提前结束训练
                        print("No optimization for a long time, auto-stopping...")
                        flag = True
                        break  # 跳出循环
                if flag:
                    break

    def evaluate(self, sess, count, test_data_count):
        sess.run(self.dev_data_op)
        all_loss = 0
        dev_count = 0
        # 1078209
        id2index = read_json(self.config.tag_path)
        index2id = {}
        for x in id2index.keys():
            index2id[id2index[x]] = x
        data_size = test_data_count // self.config.batch_size + 1

        data = []
        p_data = []
        for step in range(data_size):

            """
            h_v:字符-标签转移得分
            g_v:标签-标签转移得分
            """
            loss_train, dev_h_v, dev_g_v, tag, mask, summary = sess.run(
                [self.dev_loss, self.dev_h_v, self.dev_g_v, self.dev_tag,
                 self.dev_mask, self.summary_dev_loss])

            """
            根据h_v, g_v 计算得分最高的标签序列
            """
            # 计算预测值
            p_tag = [tf.contrib.crf.viterbi_decode(dev_h_v[index, :, :], dev_g_v)[0] for index in
                     range(dev_h_v.shape[0])]
            mask = mask.argmax(axis=1)
            p_tag = [x[:mask[i]] for i, x in enumerate(p_tag)]
            # p_tag = self.__p(dev_h_v, dev_g_v, mask)
            # 计算准确率
            """
            计算F1 值
            """
            p_tag = np.array(p_tag)
            for index, x in enumerate(mask):
                # 计算f1值
                # 统计预测分割字段以及预测标签
                """
                p_tag当前是标签的所有,把它转化为对应的标签
                例如<S_C>:1,<S_C>是实际标签，1是这个标签对应的索引
                """
                p_c = [index2id[i] for i in p_tag[index]]
                """
                分割文本,得到的p_c,p_t分别是:
                p_c:['1_2_3','4_2','5'],一个数组,每个元素是要抽取出的字段
                p_t:['O','C','O'], 对应标签
                """
                p_c, p_t = self.split(p_c)

                # 生成文本
                p_data.append('  '.join(['%s/%s' % (p_c[_], p_t[_]) for _ in range(len(p_c))]))

                """
                tag是正确的标注,做和上部分一样的处理
                """
                c = [index2id[i] for i in tag[index, :x]]
                c, t = self.split(c)
                data.append('  '.join(['%s/%s' % (c[_], t[_]) for _ in range(len(c))]))

            if not math.isnan(loss_train):
                self.log_writer.add_summary(summary, count * data_size + step)
                all_loss += loss_train
                dev_count += 1

        F1 = micro_f1(p_data, data)
        return all_loss / dev_count, F1

    def p(self, load_path, data):
        rt = []

        id2index = read_json(self.config.tag_path)
        index2id = {}
        for x in id2index.keys():
            index2id[id2index[x]] = x

        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            self.saver.restore(sess, load_path)
            w2i = read_json('./data/info/w2v/w2i.json')
            for ii, x in enumerate(tqdm(data)):
                cache = [w2i[word] if word in w2i.keys() else w2i['<NONE>'] for word in x]
                seq = []
                seq.extend(cache)
                mask = len(seq)
                split_size = len(seq) // 202 + 1
                max_length = split_size * 202
                cache = kr.preprocessing.sequence.pad_sequences([seq], max_length, value=len(w2i) - 1, padding='post',
                                                                truncating='post')
                h_v, g_v = sess.run([self.p_h_v, self.p_g_v],
                                    feed_dict={self.seq_placeholder: cache})

                p_tag = [tf.contrib.crf.viterbi_decode(h_v[index, :, :], g_v)[0] for index in
                         range(h_v.shape[0])]

                # p_tag = self.__p(h_v, g_v, [mask])
                p_tag = p_tag[0][:mask]
                p_tag = [index2id[p] for p in p_tag]
                c, p_tag = self.split(p_tag)
                cache = []
                c = [cc for cc in c if cc != '']
                for t_i, w in enumerate(c):
                    cache.append('_'.join([x[int(i)] for i in w.split('_')]) + '/' + p_tag[t_i].lower())
                rt.append('  '.join(cache))
        return rt

    def p_last(self, load_path, data, char_cache):
        rt = []

        id2index = read_json(self.config.tag_path)
        index2id = {}
        for x in id2index.keys():
            index2id[id2index[x]] = x

        ft_name = [o for o in char_cache.keys()]

        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            self.saver.restore(sess, load_path)
            w2i = read_json('./data/w2i.json')
            for ii, x in enumerate(tqdm(data)):
                cache = [w2i[word] if word in w2i.keys() else w2i['[UNK]'] for word in x]
                seq = [w2i['[CLS]']]
                seq.extend(cache)
                seq.append(w2i['[SEP]'])

                mask = len(seq) - 2
                split_size = len(seq) // 202 + 1
                max_length = split_size * 202

                # 生成特征
                ft = [[0.0 for o in ft_name]]
                for w in x[-max_length + 2:]:
                    ft_c = [0.0 for o in ft_name]
                    for i, k in enumerate(ft_name):
                        if w in char_cache[k]:
                            ft_c[i] = 1.0
                    ft.append(ft_c)

                if max_length > len(ft):
                    for i in range(max_length - len(ft)):
                        ft.append([0.0 for o in ft_name])
                ft = np.array([ft])

                cache = kr.preprocessing.sequence.pad_sequences([seq], max_length, value=0, padding='post',
                                                                truncating='post')
                h_v_c = []
                for index in range(len(cache) // 512 + 1):
                    content = cache[:, 512 * index:512 * (index + 1)]
                    ft_split = ft[:, 512 * index:512 * (index + 1), :]
                    if len(content) == 0:
                        continue
                    h_v, g_v = sess.run([self.p_h_v, self.p_g_v],
                                        feed_dict={self.seq_placeholder: content, self.ft_placeholder: ft_split})
                    h_v_c.append(h_v)
                h_v = np.concatenate(h_v_c, axis=1)

                p_tag = [tf.contrib.crf.viterbi_decode(h_v[index, :, :], g_v)[0] for index in
                         range(h_v.shape[0])]

                # p_tag = self.__p(h_v, g_v, [mask])
                p_tag = p_tag[0][:mask]
                p_tag = [index2id[p] for p in p_tag]
                rt.append(' '.join([x[i] + ',' + w.upper() for i, w in enumerate(p_tag)]))
        return rt

    def get_time_dif(self, start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))


def get_data_count(path):
    c = 0
    for x in path:
        for record in tf.python_io.tf_record_iterator(x):
            c += 1
    return c


class TCNNConfig(object):
    """CNN配置参数"""

    def __init__(self):
        self.seq_length = 202
        self.embedding_size = 384
        self.vocab_size = 17248
        self.pos_size = 12
        self.batch_size = 256
        self.learning_rate = [1e-4, 1e-3]
        self.n_head = 6

        self.print_per_batch = 100  # 每多少轮输出一次结果
        self.dev_per_batch = 500  # 多少轮验证一次

        self.train_data_path = '../data/train/20190703161437/train_pd.record'
        self.test_data_path = '../data/train/20190703161437/dev_pd.record'
        self.dev_data_path = '../data/train/20190703161437/dev_pd.record'
        self.tag_path = '../data/train/20190703161437/id2index.json'
        self.num_epochs = 200

    def get_all_number(self):
        rt = {}
        for key, value in vars(self).items():
            rt[key] = value
        return rt
