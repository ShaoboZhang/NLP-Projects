# -*- coding: utf-8 -*-
"""
@Author  : Shaobo Zhang
@DateTime: 2020/11/17
@Project : Kaikeba
@FileName: main.py
"""
import torch.utils.data as Data
from utils.preprocess import Preporcess, SQuadDataset
from utils import config
from utils.model import BiDAF

if __name__ == '__main__':
    processor = Preporcess(config)
    squad_dataset = SQuadDataset(processor, config.train_file_path)
    squad_dataloder = Data.DataLoader(squad_dataset, batch_size=config.batch_size)
    model = BiDAF(config)
    # model.to(config.device)
    for input_ids, token_type_ids, attention_mask, c_chars, q_chars, _ in squad_dataloder:
        pos1, pos2 = model(input_ids, token_type_ids, attention_mask, c_chars, q_chars)
        print(pos1.shape, pos2.shape)
        break
