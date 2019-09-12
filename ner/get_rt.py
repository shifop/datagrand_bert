from tqdm import tqdm
from ner.tools import *
from collections import Counter

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
    for index,x in enumerate(data):
        cache.append(str(index))
        if x =='O':
            o_c += 1
        if ('S' in x and len(cache) > 1) or (x == 'O' and o_c == 1 and len(cache)> 1):
            # if o_c != 1 or not (x == 'O' and o_c == 1 and len(cache)> 1):
            if not (x == 'O' and o_c == 1 and len(cache) > 1):
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

def read_data(path=['0','1','2','3','4']):
    data = []
    words = []
    for x in path:
        with open('../../raw/5/%s.txt'%(x), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                data.append(line)
                words.extend(line.split('  '))
    return "。".join(data), set(words)

id2index=read_json('../../train/20190804185806/id2index.json')
index2id = {}
for key in id2index.keys():
    index2id[id2index[key]] = key
w2i = read_json('../../w2i.json')
data = []
with open('../../raw/test.txt', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(line.strip().split('_'))

tag = []
for tag_path in ['20190813010833','20190813102935','20190813121253','20190813134150','20190813152233',
                 '20190827173613','20190827191608','20190827205024','20190827221318','20190828000946']:
    tag.append([])
    # data.append([])
    cache = read_text('./%s_result.txt'%(tag_path))
    for x in cache:
        tag[-1].append([w.split(',')[-1] for w in x.strip().split(' ')])
        # data[-1].append([w.split(',')[0] for w in x.strip().split(' ')])
# data = data[0]
rt = []
# 读取训练数据
raw_data, words = read_data()
rt = []
for ii, x in enumerate(tqdm(data)):
    if len(x) != len(tag[0][ii]):
        continue
    start = 0
    end = len(x)
    cache = []
    while(start<end):
        # 获取候选集
        rt_c = []
        for tag_c in tag:
            c, p_tag = split1(tag_c[ii][start:end])
            rt_c.append('%s/%s'%('_'.join([x[int(_)+start] for _ in c[0].split('_')]),p_tag[0].lower()))
        # 评估候选集
        if len(set(rt_c))==1:
            cache.append(rt_c[0])
            # start = int(rt_c[0][:-2].split('_')[-1])+1
            start += len(rt_c[0].split('_'))
        else:
            rt_c = Counter(rt_c)
            score = []
            rt_c_keys = [_ for _ in rt_c.keys()]
            for x_ in rt_c_keys:
                """
                计分标准
                1. 是否具有相同字段
                2. 是否具有相同上下文
                3. 候选集频率
                (X+Y)*Z+Z
                """
                X=0
                Y=0
                Z=0
                if x_ in words:
                    X=1
                if len(cache)>=1:
                    content = "%s  %s  "%("%s/%s"%('_'.join(cache[-1][:-2].split('_')[-2:]),cache[-1][-1]),x_)
                else:
                    content = "%s  " % (x_)
                if content in raw_data:
                    Y=1
                Z = rt_c[x_]
                score.append((X+Y)*Z+Z)
                # score.append(X+Y+Z)
            max_score = max(score)
            cache.append(rt_c_keys[score.index(max_score)])
            # start = int(rt_c_keys[score.index(max_score)][:-2].split('_')[-1])+1
            start += len(rt_c_keys[score.index(max_score)].split('_'))
    # rt.append('  '.join(['_'.join([x[int(index)] for index in _[:-2].split('_')])+_[-2:] for _ in cache]))
    rt.append('  '.join(cache))
    # print()



with open('./result.txt', 'w', encoding='utf-8') as f:
    for x in rt:
        f.write(x + "\n")