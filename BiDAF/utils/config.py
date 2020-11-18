# -*- coding: utf-8 -*-
"""
@Author  : Shaobo Zhang
@DateTime: 2020/11/17
@Project : Kaikeba
@FileName: config.py
@Discribe: setup configuration parameters
"""
import logging

# 日志信息
logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger()
# 数据保存路径
train_file_path = './data/squad/train-v1.1.json'
dev_file_path = './data/squad/dev-v1.1.json'
glove_path = './data/glove/glove.6B.50d.txt'

# 数据预处理
max_seq_clen = 200
max_seq_qlen = 32
max_char_len = 10
max_vocab_size = 20000
max_char_size = 1000

# 模型参数
batch_size = 16
dropout = 0.4
embed_size = 50
kernel_size = 3
hidden_size = 50

