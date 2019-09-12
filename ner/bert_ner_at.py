# coding: utf-8

from datetime import timedelta
from ner.tools import *
import tensorflow as tf
import time
from tqdm import tqdm
import tensorflow.contrib.keras as kr
from bert.modeling import *
from ner.attention import *
import logging

logging.basicConfig(level=logging.INFO)

class Attention(object):

    def __init__(self, config, split_fn):
        self.config = config
        self.split = split_fn
        self.__createModel()

    def __encode(self, cells, seq, ft, config, training=True):
        """
        使用bilstm编码
        :param seq: 待标注序列
        :param config: 相关配置信息
        :return: h(y_i|X)
        """
        # load bert embedding
        bert_config = BertConfig.from_json_file(self.config.bert_config)  # 配置文件地址。
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

        output = cells(embedding)


        with tf.variable_scope("at", reuse=tf.AUTO_REUSE):
            h_v = attn(output, "attention", config.hidden_size * 2, hparams=config)
        h_v = tf.squeeze(h_v, axis=-1)
        h_v = tf.transpose(h_v, [0,2,1])[:,1:-1,:]
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
                g_v_value = np.load(self.config.g_value)
                g_v_value = np.reshape(g_v_value, [self.config.pos_size * self.config.pos_size, 1])
                g_v = tf.get_variable('g_v', [self.config.pos_size * self.config.pos_size, 1], tf.float32,
                                      initializer=tf.constant_initializer(g_v_value))
                g_v = tf.reshape(g_v, [config.pos_size, config.pos_size])
                loss, _ = tf.contrib.crf.crf_log_likelihood(h_v, tag, mask, g_v)

        return tf.reduce_mean(-loss)

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

    def __test(self, cell, seq, ft):
        h_v = self.__encode(cell, seq, ft, self.config, False)
        with tf.name_scope("decode"):
            with tf.variable_scope("var-decode", reuse=tf.AUTO_REUSE):
                # tag-tag 得分矩阵
                g_v_value = np.load(self.config.g_value)
                g_v_value = np.reshape(g_v_value, [self.config.pos_size * self.config.pos_size, 1])
                g_v = tf.get_variable('g_v', [self.config.pos_size * self.config.pos_size, 1], tf.float32,
                                      initializer=tf.constant_initializer(g_v_value))
                g_v = tf.reshape(g_v, [self.config.pos_size, self.config.pos_size])
        return h_v, g_v

    def __train(self, cell, seq, ft, tag, tag_p2p, mask, training=True):
        h_v = self.__encode(cell, seq, ft, self.config, training)
        loss = self.__crf_loss(seq, tag, tag_p2p, h_v, mask, self.config)
        with tf.name_scope("decode"):
            with tf.variable_scope("var-decode", reuse=tf.AUTO_REUSE):
                # tag-tag 得分矩阵
                g_v_value = np.load(self.config.g_value)
                g_v_value = np.reshape(g_v_value, [self.config.pos_size * self.config.pos_size, 1])
                g_v = tf.get_variable('g_v', [self.config.pos_size * self.config.pos_size, 1], tf.float32,
                                      initializer=tf.constant_initializer(g_v_value))
                g_v = tf.reshape(g_v, [self.config.pos_size, self.config.pos_size])
        return loss, h_v, g_v

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
        def parser(record):
            features = tf.parse_single_example(record,
                                               features={
                                                   'seq': tf.FixedLenFeature([self.config.seq_length], tf.int64),
                                                   'ft': tf.FixedLenFeature([], tf.string),
                                                   'tag': tf.FixedLenFeature([self.config.seq_length], tf.int64),
                                                   'mask': tf.FixedLenFeature([self.config.seq_length-2], tf.int64)
                                               }
                                               )
            return features['seq'], features['ft'], features['tag'], features['mask']

        self.graph = tf.Graph()
        with self.graph.as_default():
            seq, ft, tag, tag_p2p, self.mask, self.train_data_op = self.__get_data(self.config.train_data_path, parser,
                                                                                   is_train=True)
            dev_seq, dev_ft, dev_tag, dev_tag_p2p, self.dev_mask, self.dev_data_op = self.__get_data(
                self.config.test_data_path, parser)

            cells = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(self.config.hidden_size,
                                                                            return_sequences=True))

            self.loss, self.h_v, self.g_v = self.__train(cells, seq, ft, tag, tag_p2p, self.mask, training=True)
            self.dev_loss, self.dev_h_v, self.dev_g_v = self.__train(cells, dev_seq, dev_ft, dev_tag, dev_tag_p2p,
                                                                     self.dev_mask, training=False)

            self.summary_train_loss = tf.summary.scalar('train_loss', self.loss)
            self.summary_dev_loss = tf.summary.scalar('dev_loss', self.dev_loss)

            # 创建用于预测的相关图
            self.seq_placeholder = tf.placeholder(shape=[1, None], dtype=tf.int32, name="seq_placeholder")
            self.ft_placeholder = tf.placeholder(shape=[1, None, 9], dtype=tf.float32, name="ft_placeholder")
            self.split_size = tf.placeholder(shape=(1), dtype=tf.int32, name="split_size")

            self.p_h_v, self.p_g_v = self.__test(cells, self.seq_placeholder, self.ft_placeholder)

            self.dev_tag = dev_tag

            tv = tf.trainable_variables()

            # 优化器
            self.optim = self.__get_train_op(tv[:103], tv[103:], self.loss)
            self.saver = tf.train.Saver(tf.trainable_variables())
            for index, x in enumerate(tf.trainable_variables()):
                logging.info('%d:%s' % (index, x))
            self.saver_v = tf.train.Saver(tf.trainable_variables()[:103])

            self.merged = tf.summary.merge_all()

    def train(self, load_path, save_path, log_path, is_reload=False):
        self.log_writer = tf.summary.FileWriter(log_path, self.graph)
        logging.info('Training and evaluating...')
        start_time = time.time()
        total_batch = 0  # 总批次
        min_loss = -1
        last_improved = 0  # 记录上一次提升批次
        test_data_count = get_data_count(self.config.test_data_path)
        train_data_count = get_data_count(self.config.train_data_path)
        size = train_data_count // self.config.batch_size if train_data_count % self.config.batch_size==0 else train_data_count // self.config.batch_size+1
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
                logging.info('Epoch:%d'%(epoch + 1))
                sess.run(self.train_data_op)
                for step in range(size):
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
                            msg = 'Iter: {0:>6}, Train Loss: {1:>6.5}, Val loss: {2:>6.5}, Val F1:{3:>6.5} Time: {4} {5}'
                            logging.info(msg.format(total_batch, all_loss / self.config.print_per_batch, dev_loss, dev_acc,
                                             time_dif, improved_str))
                        else:
                            time_dif = self.get_time_dif(start_time)
                            msg = 'Iter: {0:>6}, Train Loss: {1:>6.5}, Time: {2}'
                            logging.info(msg.format(total_batch, all_loss / self.config.print_per_batch, time_dif))
                        all_loss = 0

                    loss_train, summary, _ = sess.run([self.loss, self.summary_train_loss, self.optim])  # 运行优化
                    self.log_writer.add_summary(summary, total_batch)
                    all_loss += loss_train
                    total_batch += 1

                    if total_batch - last_improved > require_improvement:
                        # 验证集正确率长期不提升，提前结束训练
                        logging.info("No optimization for a long time, auto-stopping...")
                        flag = True
                        break  # 跳出循环
                if flag:
                    break

    def evaluate(self, sess, count, test_data_count):
        sess.run(self.dev_data_op)
        all_loss = 0
        dev_count = 0
        id2index = read_json(self.config.tag_path)
        index2id = {}
        for x in id2index.keys():
            index2id[id2index[x]] = x
        size = test_data_count // self.config.batch_size if test_data_count % self.config.batch_size==0 else test_data_count // self.config.batch_size+1

        data = []
        p_data = []
        for step in range(size):

            loss_train, dev_h_v, dev_g_v, tag, mask, summary = sess.run(
                [self.dev_loss, self.dev_h_v, self.dev_g_v, self.dev_tag,
                 self.dev_mask, self.summary_dev_loss])

            # 计算预测值
            p_tag = [tf.contrib.crf.viterbi_decode(dev_h_v[index, :, :], dev_g_v)[0] for index in
                     range(dev_h_v.shape[0])]
            mask = mask.argmax(axis=1)
            p_tag = [x[:mask[i]] for i, x in enumerate(p_tag)]

            p_tag = np.array(p_tag)
            for index, x in enumerate(mask):
                p_c = [index2id[i] for i in p_tag[index]]
                p_c, p_t = self.split(p_c)
                p_data.append('  '.join(['%s/%s' % (p_c[_], p_t[_]) for _ in range(len(p_c))]))

                c = [index2id[i] for i in tag[index, :x]]
                c, t = self.split(c)
                data.append('  '.join(['%s/%s' % (c[_], t[_]) for _ in range(len(c))]))

            if not math.isnan(loss_train):
                self.log_writer.add_summary(summary, count * size + step)
                all_loss += loss_train
                dev_count += 1

        F1 = micro_f1(p_data, data)
        return all_loss / dev_count, F1

    def p(self, load_path, data, char_cache):
        rt = []

        id2index = read_json(self.config.tag_path)
        index2id = {}
        for x in id2index.keys():
            index2id[id2index[x]] = x

        ft_name = [o for o in char_cache.keys()]

        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            self.saver.restore(sess, load_path)
            w2i = read_json('../data/raw/w2i.json')
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
                for index in range(len(cache[0]) // 512 + 1):
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
