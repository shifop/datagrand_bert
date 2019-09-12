from ner.bert_ner_dense import Dense
from ner.tools import *
import os


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

def split(data):
    rt = []
    tag = []
    cache = []
    o_c = 0
    for index,x in enumerate(data):
        cache.append(str(index))
        if x =='O':
            o_c += 1
        if ('S' in x and len(cache) > 1) or (x == 'O' and o_c == 1 and len(cache)> 1):
            if o_c != 1 or not (x == 'O' and o_c == 1 and len(cache)> 1):
                o_c = 0
            rt.append('_'.join(cache[:-1]))
            cache = [cache[-1]]
            if len(data[index-1]) > 1:
                tag.append(data[index-1][1])
            else:
                tag.append(data[index-1])
    if len(cache)!=0:
        rt.append('_'.join(cache))
        if len(data[-1])>1:
            tag.append(data[-1][1])
        else:
            tag.append(data[-1])
    return rt,tag


if __name__=='__main__':
    dirs = '20190911194226'  # 训练数据目录

    # 注明训练情况，方便以后查看
    msg = []
    now_tim = get_time()

    l_path = '../data/bert/model.ckpt'

    path = '../data/model/%s/model.ckpt' % (now_tim)

    log_path = '../data/model/%s_log/' % (now_tim)

    config = Config()
    config.pos_size = len(read_json('../data/train/ner/%s/id2index.json'%(dirs)))
    config.n_head = config.pos_size
    # config.n_head = 1
    config.set_path(dirs)
    msg.append("参数:%s" % (json.dumps(config.get_all_number())))
    msg.extend(read_text('../data/train/ner/%s/log.txt'%(dirs)))

    oj = Dense(config, split)
    if not os.path.exists('../data/model/%s' % (now_tim)):
        os.makedirs('../data/model/%s' % (now_tim))
    save_text(msg, '../data/model/%s/log.txt' % (now_tim))

    # 开始训练
    oj.train(l_path, path, log_path)