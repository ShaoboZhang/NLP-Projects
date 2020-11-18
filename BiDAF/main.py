# -*- coding: utf-8 -*-
"""
@Author  : Shaobo Zhang
@DateTime: 2020/11/17
@Project : Kaikeba
@FileName: main.py
"""
from utils.preprocess import Preporcess, SquadDataset
from utils import config
from utils.model import BiDAF
import torch.utils.data as tud

if __name__ == '__main__':
    processor = Preporcess(config)
    train_dataset = SquadDataset(processor, config.train_file_path)
    train_loader = tud.DataLoader(train_dataset, config.batch_size, shuffle=True)
    model = BiDAF(config, processor.glove_matrix)
    for (c_words, c_chars, q_words, q_chars, ans_pos) in train_loader:
        pos1, pos2 = model(c_words, c_chars, q_words, q_chars)
        print(pos1.shape, pos2.shape)
        break
    pass
