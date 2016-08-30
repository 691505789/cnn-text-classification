# encoding=utf-8

import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import jieba
import matplotlib as mpl

path = '/home/ubuntu/Documents/NLPDataSets/outputWithStop/'
# negTexts = list(open(os.path.join(path,"negText.txt")).readlines())
# posTexts = list(open(os.path.join(path,"posText.txt")).readlines())
#
# negTexts = [line.split() for line in negTexts if line.strip() != ""]
# posTexts = [line.split() for line in posTexts if line.strip() != ""]
# x_texts = negTexts + posTexts

'''
    获得数据集中句子长度出现次数最多的句子长度
    获得句子长度出现频率之和大于percentage的句子长度与句子出现频率
    返回：{"freq_text":len_count_map[0],"border_item":border_item}
    len_count_map[0] = [sentenceLength,frequence]
    border_item = [sentenceLength,frequence] 大于百分比的句子的长度与出现频率
    其中texts所有的文本
'''
def freq_factor(texts, percentage=0.85, savefile=None, drawable=False):
    len_count_map = defaultdict(int)

    # 句子长度为key,句子长度出现的频率为value
    for t in texts:
        len_count_map[len(t)] = len_count_map[len(t)] + 1

    # 将dict按value进行由大到小排序
    # key 是句子长度，value是出现次数
    len_count_list = sorted(len_count_map.items(),key=lambda d:d[1],reverse=True)

    # 得到key和频率
    keys = [item[0] for item in len_count_list]
    counts = [item[1] for item in len_count_list]
    min_length = min(keys) # 最短句子长度
    freq_length = keys[0]  # 最频繁句子长度
    max_length = max(keys) # 最长句子长度

    print("最短句子长度{},出现次数{}".format(min_length,len_count_map[min_length]))
    print("最频繁句子长度{},出现次数{}".format(freq_length, len_count_map[freq_length]))
    print("最长句子长度{},出现次数{}".format(max_length, len_count_map[max_length]))


    accept_length = []
    total = 0
    for item in len_count_list:
        # print("句子长度：{},出现次数:{}".format(item[0],item[1]))
        accept_length.append(item)
        total = total+item[1]
        if total/sum(counts) > percentage:
            # 将长度-出现频率按照长度进行排序
            accept_length = sorted(accept_length, key=lambda d:d[0], reverse=False)
            print("出现次数最多的句子长度是:{}".format(keys[0]))
            print("出现次数：{}".format(counts[0]))
            print("句子长度在{}~{}占据所有数据的{}".format(accept_length[0][0], accept_length[-1][0],percentage))
            print("句子长度为{},出现的次数为{}".format(accept_length[-1][0], accept_length[-1][1]))
            break

    if drawable:
        zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
        # 将list按key进行由大到小排序
        # [0] 是句子长度，[1]是出现次数
        len_count_list = sorted(len_count_map.items(), key=lambda d: d[0], reverse=True)
        keys = [item[0] for item in len_count_list]
        counts = [item[1] for item in len_count_list]


        # 将所有的数据的句子长度——出现频率
        # 选取的百分比可接受的数据（此处为2/3）的句子长度——出现频率
        # 绘制在同一张图中
        plt.plot(keys,counts,color="blue",linewidth=1.0, linestyle="-",label="total")
        plt.plot([item[0] for item in accept_length],
                    [item[1] for item in accept_length],
                 color="red",linewidth=3.5, linestyle="-",label="accept2/3")
        plt.legend(loc='upper right', frameon=False)

        plt.xlabel('句子长度', fontproperties=zhfont)
        plt.ylabel('出现次数', fontproperties=zhfont)
        # 标注两个重要的可接受的点
        # 最短长度  最长长度 最频繁出现的句子长度
        plt.annotate('({},{})'.format(accept_length[0][0], accept_length[0][1]),
                     xy=(accept_length[0][0], accept_length[0][1]), xycoords='data',
                     xytext=(+10, +30), textcoords='offset points', fontsize=10,
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        plt.annotate('({},{})'.format(accept_length[-1][0], accept_length[-1][1]),
                     xy=(accept_length[-1][0], accept_length[-1][1]), xycoords='data',
                     xytext=(+20, +30), textcoords='offset points', fontsize=10,
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        plt.annotate('({},{})'.format(freq_length, len_count_map[freq_length]),
                     xy=(freq_length, len_count_map[freq_length]), xycoords='data',
                     xytext=(+20, +30), textcoords='offset points', fontsize=10,
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

        if savefile!=None:
            plt.savefig("./figures/"+savefile, dpi=72)
        plt.show()

    return accept_length
'''
    加载原始数据和标签
    原始数据以单一文件的形式存放在文件夹中
'''
def load_data_from_dir(dir_path,data_dir):
    # 目录
    neg_dir = os.path.join(dir_path, data_dir[0])
    pos_dir = os.path.join(dir_path, data_dir[1])
    # 文件列表
    neg_file_list = os.listdir(neg_dir)
    pos_file_list = os.listdir(pos_dir)
    # 去除文本内的换行符
    neg_examples = []
    for f in neg_file_list:
        lines_list = [line.strip() for line in open(os.path.join(neg_dir,f)).readlines() if line.strip()!=""]
        file_one_line=""
        for line in lines_list:
            file_one_line = file_one_line+line
        neg_examples.append(file_one_line)

    pos_examples = []
    for f in pos_file_list:
        lines_list = [line.strip() for line in open(os.path.join(pos_dir, f)).readlines() if line.strip()!=""]
        file_one_line = ""
        for line in lines_list:
            file_one_line = file_one_line+line
        pos_examples.append(file_one_line)


    neg_labels = [[1,0] for _ in neg_examples]
    pos_labels = [[0,1] for _ in pos_examples]
    # 按行串联
    y_labels = np.concatenate([neg_labels,pos_labels], axis=0)

    return [neg_examples,pos_examples, y_labels]

'''
    将原始文本进行中文分词
    使用中文分词工具jieba
'''
def get_cut_data(texts):
    texts = [jieba.cut(t) for t in texts]
    return texts
'''
    写文件
'''
def writer(file_name, texts):
    with open(file_name,"w") as f:
        for t in texts:
            f.write(" ".join(t) + "\n")

'''
    获取分词后的neg和pos文件
    数据存放在两个文件中
'''
def load_data_and_labels(data_dir,files):
    neg_file = files[0]
    pos_file = files[1]

    neg_examples = list(open(os.path.join(data_dir,neg_file)).readlines())
    neg_examples = [line.strip().split() for line in neg_examples]

    pos_examples = list(open(os.path.join(data_dir,pos_file)).readlines())
    pos_examples = [line.strip().split()for line in pos_examples]

    x_texts = pos_examples + neg_examples

    neg_labels = [[1,0] for _ in neg_examples]
    pos_labels = [[0,1] for _ in pos_examples]

    y_labels = np.concatenate([pos_labels,neg_labels], axis=0)

    return x_texts,pos_examples,neg_examples,y_labels

'''
    首先的第一步，将原始分散的单文件分词后写入同一个文件中
    将data_origin中分散的零散文件分词整合后写入data_process
    的两个文件中
'''
def trans_files_to_file():
    # 首先的第一步，将原始分散的单文件分词后写入同一个文件中
    dir_path = './data_origin/Jingdong'
    data_dir = ["neg", "pos"]
    # 获取原始的评论数据
    neg_examples, pos_examples, y_labels = load_data_from_dir(dir_path=dir_path, data_dir=data_dir)
    # 对原始数据进行分词
    neg_examples = get_cut_data(neg_examples)
    pos_examples = get_cut_data(pos_examples)
    # 将分散的单元文件内容分词后写到同一个文件中
    writer("./data_process/jd/reviews.neg", neg_examples)
    writer("./data_process/jd/reviews.pos", pos_examples)


if __name__=="__main__":
    dir_path = './data_process/jd'
    # 第一个文件必须是消极，第二个是积极
    files = ["reviews.neg", "reviews.pos"]

    # 读取已经分词后的数据
    x_texts, pos_examples, neg_examples, y_labels = \
        load_data_and_labels(dir_path,files)

    freq_factor(x_texts,percentage=2/3,drawable=True)
    freq_factor(pos_examples, percentage=2/3,drawable=True)
    freq_factor(neg_examples, percentage=2/3,drawable=True)
    # trans_files_to_file()





