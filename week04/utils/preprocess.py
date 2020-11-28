# -*- coding: utf-8 -*-
"""
@Author  : Shaobo Zhang
@DateTime: 2020/11/17
@Project : Kaikeba
@FileName: preprocess.py
@Discribe: Preprocess Squad dataset and generate data for model
"""

import json, torch

import torch.utils.data as Data
from nltk import word_tokenize
from transformers import BertTokenizer
from tqdm import tqdm


class Preporcess:
    def __init__(self, args):
        # 获取数据处理相关参数
        self.max_seq_clen = args.max_seq_clen
        self.max_seq_qlen = args.max_seq_qlen
        self.max_char_len = args.max_char_len
        self.max_vocab_size = args.max_vocab_size
        self.logger = args.logger
        # 生成字符集与映射词典
        self.chars, self.char2idx = self.load_dataset(args.train_file_path)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
        args.max_char_size = len(self.chars)

    # 生成模型的输入
    def get_dataset(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        self.logger.info("数据收集中...")
        contexts, questions, ans_poses = [], [], []
        for context, question, answer, answer_start in self.iter_cqa(dataset):
            contexts.append(context)
            questions.append(question)
            answer_end = answer_start + len(answer)
            ans_poses.append([answer_start, answer_end])
        return contexts, questions, ans_poses

    def load_dataset(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        self.logger.info("生成字符表...")
        chars = set()
        for context, question, _, _ in self.iter_cqa(dataset):
            chars |= set(context) | set(question)
        chars = ['pad', 'unk'] + sorted(list(chars))
        char2idx = {char: idx for idx, char in enumerate(chars)}
        return chars, char2idx

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

    def word_encode(self, context, question):
        # truncate context with padding
        c_words = [word.lower() for word in word_tokenize(context)]
        c_words = c_words[:self.max_seq_clen] + ['[PAD]'] * (self.max_seq_clen - len(c_words))
        # truncate context with padding
        q_words = [word.lower() for word in word_tokenize(question)]
        q_words = q_words[:self.max_seq_qlen] + ['[PAD]'] * (self.max_seq_qlen - len(q_words))
        encoded_sent = self.tokenizer.encode_plus(c_words, q_words)
        input_ids = encoded_sent['input_ids']
        token_type_ids = encoded_sent['token_type_ids']
        attention_mask = encoded_sent['attention_mask']
        return map(torch.tensor, (input_ids, token_type_ids, attention_mask))

    def char_encode(self, words, max_seq_len):
        chars_vec = []
        words = [word.lower() for word in word_tokenize(words)][:max_seq_len]
        words += [0] * (max_seq_len - len(words))
        for word in words:
            if word == 0:
                chars_vec.append([0] * self.max_char_len)
            else:
                chars = [self.char2idx.get(char, 1) for char in word][:self.max_char_len]
                chars += [0] * (self.max_char_len - len(chars))
                chars_vec.append(chars)
        return torch.tensor(chars_vec)


class SQuadDataset(Data.Dataset):
    def __init__(self, processor: Preporcess, file_path):
        self.processor = processor
        self.contexts, self.questions, self.ans_poses = processor.get_dataset(file_path)
        pass

    def __getitem__(self, item):
        context, question, ans_pos = self.contexts[item], self.questions[item], self.ans_poses[item]
        input_ids, token_type_ids, attention_mask = self.processor.word_encode(context, question)
        c_chars = self.processor.char_encode(context, self.processor.max_seq_clen)
        q_chars = self.processor.char_encode(question, self.processor.max_seq_qlen)
        ans_pos = torch.tensor(ans_pos)
        return input_ids, token_type_ids, attention_mask, c_chars, q_chars, ans_pos

    def __len__(self):
        return len(self.ans_poses)
