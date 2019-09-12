import json
import datetime
import random
import numpy as np

def read_text(path):
    rt = []
    with open(path, 'r', encoding='utf-8') as f:
        for x in f:
            rt.append(x)
        return rt


def save_text(data,path):
    with open(path, 'w', encoding='utf-8') as f:
        for x in data:
            f.write(x+'\n')

def get_time():
    now_time = datetime.datetime.now()
    return now_time.strftime('%Y%m%d%H%M%S')


def read_json(path):
    with open(path,'r',encoding='utf-8') as f:
        data = json.loads(f.read())
    return data


def save_json(data, path):
    with open(path,'w',encoding='utf-8') as f:
        f.write(json.dumps(data,ensure_ascii=False))


def micro_f1(sub_lines, ans_lines, split = ' '):
    correct = []
    total_sub = 0
    total_ans = 0
    for sub_line, ans_line in zip(sub_lines, ans_lines):
        sub_line = set(str(sub_line).split(split))
        ans_line = set(str(ans_line).split(split))
        c = sum(1 for i in sub_line if i in ans_line) if sub_line != {''} else 0
        total_sub += len(sub_line) if sub_line != {''} else 0
        total_ans += len(ans_line) if ans_line != {''} else 0
        correct.append(c)
    p = np.sum(correct) / total_sub if total_sub != 0 else 0
    r = np.sum(correct) / total_ans if total_ans != 0 else 0
    f1 = 2*p*r / (p + r) if (p + r) != 0 else 0
    return f1

if __name__=='__main__':
    train = read_text('../data/raw/train.txt')
    train = [x.strip() for x in train]
    size = len(train)//5
    cache = []
    random.shuffle(train)

    for index in range(5):
        cache.append(train[index*size:(index+1)*size])

    for index in range(5):
        save_text(cache[index],'../data/raw/5/%s.txt'%(index))