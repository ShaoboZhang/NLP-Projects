# -*- coding: utf-8 -*-
"""
@Author  : Shaobo Zhang
@DateTime: 2020/11/17
@Project : Kaikeba
@FileName: preprocess.py
@Discribe: Preprocess Squad dataset and generate data for model
"""

import torch.utils.data as tud
import numpy as np
import json
import torch

from tqdm import tqdm
from nltk import word_tokenize


class Preporcess:
    def __init__(self, args):
        # 获取数据处理相关参数
        self.max_seq_clen = args.max_seq_clen
        self.max_seq_qlen = args.max_seq_qlen
        self.max_char_len = args.max_char_len
        self.max_vocab_size = args.max_vocab_size
        self.logger = args.logger
        # 生成字符集与单词集及嵌入矩阵
        self.chars, self.char2idx = self.load_dataset(args.train_file_path)
        self.words, self.word2idx, self.glove_matrix = self.load_glove(args.glove_path, args.max_vocab_size)
        args.max_char_size = len(self.chars)

    # 生成模型的输入
    def get_dataset(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        self.logger.info("数据收集中...")
        c_words, c_chars, q_words, q_chars, ans_pos = [], [], [], [], []
        for context, question, answer, answer_start in self.iter_cqa(dataset):
            context = self.word_encode(context, self.max_seq_clen)
            c_words.append(context)
            c_chars.append([self.char_encode(word, self.max_char_len) for word in context])
            question = self.word_encode(question, self.max_seq_qlen)
            q_words.append(question)
            q_chars.append([self.char_encode(word, self.max_char_len) for word in question])
            answer_end = answer_start + len(answer)
            ans_pos.append((answer_start, answer_end))
        return map(torch.tensor, (c_words, c_chars, q_words, q_chars, ans_pos))

    def load_dataset(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        self.logger.info("生成字符表...")
        chars = set()
        for context, question, _, _ in self.iter_cqa(dataset):
            chars |= set(context) | set(question)
        chars = ['pad'] + sorted(list(chars))
        char2idx = {char: idx for idx, char in enumerate(chars)}
        return chars, char2idx

    def load_glove(self, file_path, vocab_size):
        words = ['pad', 'unk']
        word_embed = [[0] * 50, [0] * 50]
        self.logger.info("加载Glove词向量...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                word, coefs = line.strip().split(maxsplit=1)
                if word == 'pad' or word == 'unk':
                    continue
                words.append(word)
                word_embed.append(np.fromstring(coefs, sep=' '))
                if len(words) == vocab_size:
                    break
        word2idx = {word: idx for idx, word in enumerate(words)}
        return words, word2idx, torch.tensor(word_embed, dtype=torch.float)

    @staticmethod
    def iter_cqa(dataset):
        for data in tqdm(dataset['data']):
            for para in data['paragraphs']:
                context = para['context']
                for qa in para['qas']:
                    # qid = qa['id']
                    question = qa['question']
                    for answers in qa['answers']:
                        answer = answers['text']
                        answer_start = answers['answer_start']
                        yield context, question, answer, answer_start

    def char_encode(self, word_idx, max_len):
        chars = [0] * max_len
        # 如果对应word是'pad'或者'unk'，则直接返回
        if word_idx == 0 or word_idx == 1:
            return chars
        word = self.words[word_idx][:max_len]
        for idx, char in enumerate(word):
            chars[idx] = self.char2idx.get(char)
        return chars

    def word_encode(self, sent, max_len):
        words = [self.word2idx.get(word.lower(), 1) for word in word_tokenize(sent)][:max_len]
        words += [0] * (max_len - len(words))
        return words


class SquadDataset(tud.Dataset):
    def __init__(self, processor: Preporcess, file_path):
        super(SquadDataset, self).__init__()
        self.c_words, self.c_chars, self.q_words, self.q_chars, self.ans_pos = processor.get_dataset(file_path)

    def __getitem__(self, item):
        return self.c_words[item], self.c_chars[item], self.q_words[item], self.q_chars[item], self.ans_pos[item]

    def __len__(self):
        return len(self.ans_pos)
