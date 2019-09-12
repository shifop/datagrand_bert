from ner.bert_ner_at import Attention
from ner.tools import *


class Config(object):
    """配置参数"""

    def __init__(self):
        self.seq_length = 202
        self.hidden_size = 300
        self.pos_size = 12
        self.batch_size = 16
        self.learning_rate = [5e-5, 1e-3]
        self.n_head = 6
        self.bert_config = "../data/config/config.json"
        self.g_value = "../data/raw/Z.npy"

        self.print_per_batch = 100  # 每多少轮输出一次结果
        self.dev_per_batch = 500  # 多少轮验证一次

        self.train_data_path = ['../data/train/ner/%s/'+'%s.record'%(str(x)) for x in [0,1,2,3]]
        self.test_data_path = ['../data/train/ner/%s_nag/'+'%d.record'%(x) for x in [4]]
        self.dev_data_path = ['../data/train/ner/%s_nag/'+'%d.record'%(x) for x in [4]]
        self.tag_path = '../data/train/ner/%s_nag/id2index.json'
        self.num_epochs = 8

    def get_all_number(self):
        rt = {}
        for key,value in vars(self).items():
            rt[key] = value
        return rt

    def set_path(self, path):
        self.train_data_path = [x % (path) for x in self.train_data_path]
        self.test_data_path = [x % (path) for x in self.test_data_path]
        self.dev_data_path = [x % (path) for x in self.dev_data_path]
        self.tag_path = self.tag_path % (path)

def process_data1(datas, is_ag=True):
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

def split1(data):
    """
    将标记分离，适用于
    “将A，B，C标签分为<*_S>,<*_O>,<*_E>三种，分别代表起始中间，结尾，由于最小字段仅1字，单字字段使用<*_S>标注，O标签统一使用<O>代表”
    :param data:
    :return:
    """
    rt = []
    tag = []
    cache = []
    o_c = 0
    # 遍历句子，提取字段
    for index,x in enumerate(data):
        # 将当前字符加入缓存区，备用
        cache.append(str(index))
        if x =='O':
            o_c += 1
        # 假如当前字符的标注是‘O’，或者带‘S’，说明缓存区中是完整的字段
        if ('S' in x and len(cache) > 1) or (x == 'O' and o_c == 1 and len(cache)> 1):
            if o_c != 1:
                o_c = 0
            # 将缓存区的字符用‘_’连接，保存
            rt.append('_'.join(cache[:-1]))
            # 清空缓存区，保留最后一个字符
            cache = [cache[-1]]
            # 对不同类型的标签做不同处理后保存
            if len(data[index-1]) > 1:
                tag.append(data[index-1][1])
            else:
                tag.append(data[index-1])
    # 有些情况下，到字符最后一个也遇不到边界，这段代码处理这种情况
    if len(cache)!=0:
        rt.append('_'.join(cache))
        if len(data[-1])>1:
            tag.append(data[-1][1])
        else:
            tag.append(data[-1])
    return rt,tag

def read_txt(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(line.strip())
    return data


if __name__=='__main__':
    # 生成标签统计
    data = [read_txt('../data/raw/ner/%s.txt' % (str(x))) for x in range(5)]
    pd, chars, split_datas, all_tag, process_msg = process_data1(data)

    char_cache = {"<B_E>": set(), "<A_E>": set(), "<C_O>": set(), "<A_S>": set(), "<B_S>": set(),
                  "<B_O>": set(),
                  "<A_O>": set(), "<C_E>": set(), "<C_S>": set()}
    for i, x in enumerate(chars):
        for k in char_cache.keys():
            char_cache[k] = char_cache[k] | x[k]

    word_index = read_json('../data/raw/w2i.json')
    data = []
    with open('../data/raw/test.txt', 'r', encoding='utf-8') as f:
        for line in f:
            data.append(line.strip().split('_'))

    for load_path in ['20190911201419']:
        dirs = '20190911194226'

        config = Config()
        config.pos_size = len(read_json('../data/train/ner/%s/id2index.json' % (dirs)))
        config.n_head = config.pos_size
        config.set_path(dirs)
        oj = Attention(config, split1)
        path = '../data/model/%s/model.ckpt'%(load_path)

        rt = oj.p(path, data, char_cache)
        with open('../data/rt/%s_result.txt'%(load_path), 'w', encoding='utf-8') as f:
            for x in rt:
                f.write(x + "\n")