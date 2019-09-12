import tensorflow as tf
from tqdm import tqdm
import numpy as np
import tensorflow.contrib.keras as kr
from ner.tools import *
import os

"""预测"""

def save_as_record(path, data, id2index, vocab_map, max_length, char_cache):
    train_writer = tf.python_io.TFRecordWriter(path)
    padding_value=vocab_map['[PAD]']
    ft_name = [o for o in char_cache.keys()]
    for x in tqdm(data):
        tag = x['tag']
        cache = x['content']
        cache = [vocab_map[word] if word in vocab_map.keys() else vocab_map['[UNK]'] for word in cache]
        tag = [id2index[t] for t in tag]

        # 生成特征
        ft = [[0.0 for o in ft_name]]
        for w in x['content'][-max_length+2:]:
            ft_c = [0.0 for o in ft_name]
            for i, k in enumerate(ft_name):
                if w in char_cache[k]:
                    ft_c[i] = 1.0
            ft.append(ft_c)

        if max_length>len(ft):
            for i in range(max_length-len(ft)):
                ft.append([0.0 for o in ft_name])

        label = []
        seq = [vocab_map['[CLS]']]
        seq.extend(cache[-max_length+2:])
        label.extend(tag[-max_length+2:])
        seq.append(vocab_map['[SEP]'])

        length = len(seq)
        mask = [0 for x in range(max_length-2)]
        mask[length - 3] = 1
        cache = kr.preprocessing.sequence.pad_sequences([seq], max_length, value=padding_value, padding='post',
                                                        truncating='post')
        label = kr.preprocessing.sequence.pad_sequences([label], max_length, value=0, padding='post',
                                                        truncating='post')

        features = tf.train.Features(feature={
            'seq': tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(cache[0].tolist(), np.int64))), # 文本内容
            'ft': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(ft, np.float32).tostring()])),
            'mask': tf.train.Feature(int64_list=tf.train.Int64List(value=mask)), # 用于计算loss
            'tag': tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(label[0].tolist(), np.int64)))}) # 预测标签
        example = tf.train.Example(features=features)
        train_writer.write(example.SerializeToString())
    train_writer.close()


def read_txt(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(line.strip())
    return data

def process(datas, is_ag=True):
    """
    将A，B，C标签分为<*_S>,<*_O>,<*_E>三种，分别代表起始中间，结尾，由于最小字段仅1字，单字字段使用<*_S>标注
    O标签统一使用<O>代表
    :param data: 待定处理数据
    :return:处理好的数据[{'content': [], 'tag': []}], 标签列表
    """
    msg = "将A，B，C标签分为<*_S>,<*_O>,<*_E>三种，分别代表起始中间，结尾，由于最小字段仅1字" \
          "，单字字段使用<*_S>标注，O标签统一使用<O>代表"
    rt = []
    all_tag = set()
    split_data = []
    split_data_o = []
    char = []
    # 处理数据的同时保持随机切割的数据
    for index, data in enumerate(datas):
        rt.append([])
        split_data.append([])
        split_data_o.append([])
        char_cache = {"<B_E>": set(), "<A_E>": set(), "<C_O>": set(), "<A_S>": set(), "<B_S>": set(), "<B_O>": set(),
                      "<A_O>": set(), "<C_E>": set(), "<C_S>": set()}

        for i, x in enumerate(data):
            cache = x.split('  ')
            sd = []
            if is_ag:
                # 如果全部为O，另外存储
                if len(set([w[-1] for w in cache])) == 1:
                    split_data_o[-1].extend(cache)
                else:
                    for x in cache:
                        xl = len(x.split('_'))
                        if x[-1]=='o' and xl>5:
                            sd.append('_'.join(x.split('_')[:xl//2])+'/o')
                            split_data[-1].append('  '.join(sd))
                            sd = ['_'.join(x.split('_')[xl//2:])]
                        else:
                            sd.append(x)
            node = {'content': [], 'tag': []}
            for content in cache:
                tag = content[-1].upper()
                content = content[:-2]
                content = content.split('_')
                node['content'].extend(content)
                if tag != 'O':
                    all_tag.add('<%s_S>' % tag)
                    all_tag.add('<%s_O>' % tag)
                    all_tag.add('<%s_E>' % tag)
                    if len(content) >= 2:
                        node['tag'].append('<%s_S>' % tag)
                        char_cache['<%s_S>' % tag].add(content[0])
                        node['tag'].extend(['<%s_O>' % tag for x in range(len(content) - 2)])
                        char_cache['<%s_O>' % tag].update(content[1:-1])
                        node['tag'].append('<%s_E>' % tag)
                        char_cache['<%s_E>' % tag].add(content[-1])
                    else:
                        node['tag'].append('<%s_S>' % tag)
                        char_cache['<%s_S>' % tag].add(content[0])
                else:
                    all_tag.add(tag)
                    node['tag'].extend([tag for x in range(len(content))])
            rt[index].append(node)
        char.append(char_cache)
    return rt, char, [split_data,split_data_o], list(all_tag), msg

def get_ext_data(split_datas, size = 2550):
    rt = []
    for split_data in split_datas:
        rt.append([])
        split_data_c = list(set(split_data))
        for x in tqdm(range(size)):
            length = 0
            cache = []
            while length < 150:
                select_w = split_data_c[random.randint(0, len(split_data_c) - 1)]
                if select_w.count('_')>202:
                    c_tag = select_w[-2:]
                    select_w = '_'.join(select_w[:-2].split('_')[:random.randint(0, 202)])
                    select_w += c_tag
                length += select_w.count('_') +1
                if length > 202:
                    continue
                if len(cache) != 0:
                    if len(select_w.split('  ')) == 1:
                        cache[-1] = cache[-1][:-2] + select_w
                    else:
                        if select_w.split('  ')[0][-1] == 'o':
                            cache[-1] = cache[-1][:-2] + '_' + select_w
                        else:
                            cache.append(select_w)
                else:
                    cache.append(select_w)
            rt[-1].append('  '.join(cache))
    return rt

if __name__=='__main__':

    msg = ["等比例分割为5份"]
    max_length = 202

    # 读取词典，调用训练词向量相关代码时会生成
    w2i = read_json('../data/raw/w2i.json')
    # 读取训练数据，数据已事先打乱后平均分割为5份
    data= [read_txt('../data/raw/ner/%d.txt'%(x)) for x in range(5)]
    pd, chars, split_datas, all_tag, process_msg = process(data)

    split_datas, split_data_o = split_datas
    msg.append(process_msg)

    data_ag = []
    data_ag.extend(get_ext_data(split_datas, 2550))
    for index,ext in enumerate(get_ext_data(split_data_o, size=850)):
        data_ag[index].extend(ext)
    data_ag, _,  _, _, _ = process(data_ag)

    id2index = read_json('../data/raw/id2index.json')

    print('开始保存数据')
    now_time = get_time()
    if not os.path.exists('../data/train/ner/%s' % (now_time)):
        os.makedirs('../data/train/ner/%s' % (now_time))
    for index,data in enumerate(pd):
        cache = []
        cache.extend(data)
        # cache.extend(data_ag[index])
        random.shuffle(cache)
        # 生成各类字段字符集合
        char_cache = {"<B_E>": set(), "<A_E>": set(), "<C_O>": set(), "<A_S>": set(), "<B_S>": set(),
                      "<B_O>": set(),
                      "<A_O>": set(), "<C_E>": set(), "<C_S>": set()}
        for i, x in enumerate(chars):
            for k in char_cache.keys():
                char_cache[k] = char_cache[k] | x[k]
        save_json(data, '../data/train/ner/%s/%d.json' % (now_time, index))
        save_as_record('../data/train/ner/%s/%d.record' % (now_time, index), cache, id2index, w2i, max_length, char_cache)
    save_text(msg, '../data/train/ner/%s/log.txt' % (now_time))
    save_json(id2index, '../data/train/ner/%s/id2index.json' % (now_time))
