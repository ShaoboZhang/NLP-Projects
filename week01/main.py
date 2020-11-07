# -*- coding: utf-8 -*-
"""
@Author: Shaobo Zhang
@Description: Generate vocab-frequency dictionary for Squad and Dureader dataset
"""

import json, jieba
from tqdm import tqdm
from collections import Counter
from nltk.tokenize import word_tokenize


def read_squad(file_path):
    words_count = Counter()
    with open(file_path) as f:
        datas = json.load(f)['data']
    print("正在生成Squad字典...")
    for data in tqdm(datas):
        for para in data['paragraphs']:
            context = para['context']
            # 文本为英文，使用nltk进行分词
            words = word_tokenize(context)
            words_count.update(words)
    with open("squad_dict.txt", 'w', encoding='utf-8') as f:
        for word, freq in words_count.most_common():
            f.write(word + f"\t{freq}\n")
    print("Squad字典已生成")


def read_dureder(file_path):
    words_count = Counter()
    with open(file_path, encoding='utf-8') as f:
        datas = json.load(f)['data']
    print("正在生成Dureader字典...")
    for data in tqdm(datas):
        for para in data['paragraphs']:
            context = para['context']
            # 文本为中文，使用jieba进行分词
            words = jieba.lcut(context)
            words_count.update(words)
    with open("dureader_dict.txt", 'w', encoding='utf-8') as f:
        for word, freq in words_count.most_common():
            f.write(word + f"\t{freq}\n")
    print("Dureader字典已生成")


if __name__ == '__main__':
    squad_file = "./datas/train-v2.0.json"
    dureader_file = "./datas/dureader_robust-data/train.json"
    read_squad(squad_file)
    read_dureder(dureader_file)
