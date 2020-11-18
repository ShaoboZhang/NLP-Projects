# -*- coding: utf-8 -*-
"""
@Author  : Shaobo Zhang
@DateTime: 2020/11/17
@Project : Kaikeba
@FileName: model.py
@Discribe: 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiDAF(nn.Module):
    def __init__(self, args, glove_matrix=None):
        super(BiDAF, self).__init__()
        self.args = args
        ################################ Embedding Layer ################################
        # Char Embedding with CNN
        self.char_embed = nn.Embedding(args.max_char_size, args.embed_size)
        nn.init.uniform_(self.char_embed.weight, 0, 0.001)
        self.char_conv = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Conv1d(args.embed_size, args.embed_size, args.kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(args.max_char_len - args.kernel_size + 1)
        )

        # Word Embedding
        # self.word_emb = nn.Embedding(args.max_vocab_size, args.embed_size)
        self.word_emb = nn.Embedding.from_pretrained(glove_matrix, freeze=True)

        # Highway Netword
        for i in range(2):
            setattr(self, f'highway_net{i}',
                    nn.Sequential(nn.Linear(2 * args.embed_size, 2 * args.hidden_size), nn.ReLU()))
            setattr(self, f'highway_gate{i}',
                    nn.Sequential(nn.Linear(2 * args.embed_size, 2 * args.hidden_size), nn.Sigmoid()))

        ############################# Contexual Embed Layer #############################
        self.context_lstm = nn.LSTM(input_size=args.hidden_size * 2,
                                    hidden_size=args.hidden_size,
                                    bidirectional=True,
                                    batch_first=True)

        ################################ Attention Layer ################################
        self.attn_c = nn.Linear(args.hidden_size * 2, 1)
        self.attn_q = nn.Linear(args.hidden_size * 2, 1)
        self.attn_cq = nn.Linear(args.hidden_size * 2, 1)

        ################################ Modeling  Layer ################################
        self.model_lstm = nn.LSTM(input_size=args.hidden_size * 8,
                                  hidden_size=args.hidden_size,
                                  num_layers=2,
                                  batch_first=True,
                                  dropout=args.dropout,
                                  bidirectional=True)

        ################################# Output Layer #################################
        # start position
        self.pos1_g = nn.Linear(args.hidden_size * 8, 1)
        self.pos1_m = nn.Linear(args.hidden_size * 2, 1)
        # end position
        self.output_lstm = nn.LSTM(input_size=args.hidden_size * 2, hidden_size=args.hidden_size,
                                   bidirectional=True, batch_first=True)
        self.pos2_g = nn.Linear(args.hidden_size * 8, 1)
        self.pos2_m = nn.Linear(args.hidden_size * 2, 1)

    def forward(self, c_words, c_chars, q_words, q_chars):
        def char_embed_layer(x):
            """
            :param x: A batch of chars with shape (batch_sz, seq_len, char_len)
            :return: (batch_sz, seq_len, embed_sz)
            """
            batch_sz, seq_len, char_len = x.shape
            # (batch_sz, seq_len, char_len) -> (batch_sz, seq_len, char_len, embed_sz)
            char_embed = self.char_embed(x)
            # (batch_sz, seq_len, char_len, embed_sz) -> (batch_sz, seq_len, embed_sz, char_len)
            char_embed = char_embed.transpose(-1, -2)
            # (batch_sz, seq_len, embed_sz, char_len) -> (batch_sz*seq_len, embed_sz, char_len)
            char_embed = char_embed.view(-1, self.args.embed_size, char_len)
            # (batch_sz*seq_len, embed_sz, char_len) -> (batch_sz*seq_len, embed_sz, conv_len) -> (batch_sz*seq_len, embed_sz, 1)
            char_conv = self.char_conv(char_embed)
            # (batch_sz * seq_len, embed_sz) -> (batch_sz, seq_len, embed_sz)
            output = char_conv.view(batch_sz, seq_len, -1).contiguous()
            return output

        def highway_network(x1, x2):
            """
            :param x1: char embedding with cnn (batch_sz, seq_len, embed_sz)
            :param x2: word embedding (batch_sz, seq_len, embed_sz)
            :return: (batch_sz, seq_len, 2*hidden_size)
            """
            x = torch.cat([x1, x2], dim=-1)
            for i in range(2):
                h = getattr(self, f'highway_net{i}')(x)
                g = getattr(self, f'highway_gate{i}')(x)
                x = g * h + (1 - g) * x
            # (batch_sz, seq_len, 2*hidden_size)
            return x

        def attn_flow_layer(c, q):
            """
            :param c: context (batch_sz, c_len, 2*hidden_sz)
            :param q: question (batch_sz, q_len, 2*hidden_sz)
            :return: (batch_sz, c_len, 8*hidden_size)
            """
            c_len = c.shape[1]
            q_len = q.shape[1]

            # compute similarity matrix
            # (batch_sz, c_len, 2 * hidden_sz) -> (batch_sz, c_len, q_len, 2*hidden_sz)
            c_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1)
            # (batch_sz, q_len, 2 * hidden_sz) -> (batch_sz, c_len, q_len, 2*hidden_sz)
            q_tiled = q.unsqueeze(1).expand(-1, c_len, -1, -1)
            # (batch_sz, c_len, q_len, 2*hidden_sz) -> (batch_sz, c_len, q_len, 1) -> (batch_sz, c_len, q_len)
            cq = self.attn_cq(c_tiled * q_tiled).squeeze()
            # (batch_sz, c_len, 2*hidden_sz) -> (batch_sz, c_len, 1) -> (batch_sz, c_len, q_len)
            h = self.attn_c(c).expand(-1, -1, q_len)
            # (batch_sz, q_len, 2*hidden_sz) -> (batch_sz, q_len, 1) -> (batch_sz, q_len, c_len)
            u = self.attn_q(q).expand(-1, -1, c_len)
            # (batch_sz, c_len, q_len)
            s = cq + h + u.transpose(1, 2)  # similarity matrix

            # context2query attention
            # (batch_sz, c_len, q_len)
            a = F.softmax(s, dim=-1)
            # (batch_sz, c_len, q_len) * (batch_sz, q_len, 2*hidden_sz) -> (batch_sz, c_len, 2*hidden_sz)
            c2q_attn = torch.bmm(a, q)

            # query2context attention
            # (batch_sz, c_len, q_len) -> (batch_sz, c_len)
            b = torch.max(s, dim=-1)[0]
            # (batch_sz, c_len) -> (batch_sz, 1, c_len)
            b = F.softmax(b, dim=-1).unsqueeze(1)
            # (batch_sz, 1, c_len) * (batch_sz, c_len, 2*hidden_sz) -> (batch_sz, 1, 2*hidden_sz) -> (batch_sz, c_len, 2*hidden_sz)
            q2c_attn = torch.bmm(b, c).expand(-1, c_len, -1)

            # (batch_sz, c_len, 8*hidden_sz)
            output = torch.cat([q2c_attn, c2q_attn, c * c2q_attn, c * q2c_attn], dim=-1)
            return output

        def output_layer(g, m):
            """
            :param g: output of attention flow layer (batch_sz, c_len, 8*hidden_sz)
            :param m: output of modeling lstm layer (batch_sz, c_len, 2*hidden_sz)
            :return: pos1: (batch_sz, c_len), pos2: (batch_sz, c_len)
            """
            # (batch_sz, c_len, 2*hidden_sz) -> # (batch_sz, c_len)
            pos1 = (self.pos1_g(g) + self.pos1_m(m)).squeeze()
            # (batch_sz, c_len, 2*hidden_sz)
            m2 = self.output_lstm(m)[0]
            # (batch_sz, c_len, 2*hidden_sz) -> # (batch_sz, c_len)
            pos2 = (self.pos2_g(g)+self.pos2_m(m2)).squeeze()
            return pos1, pos2

        # c_words, c_chars, q_words, q_chars = batch_data
        # char embedding with cnn
        c_char_embed = char_embed_layer(c_chars)
        q_char_embed = char_embed_layer(q_chars)

        # word embedding
        c_word_embed = self.word_emb(c_words)
        q_word_embed = self.word_emb(q_words)

        # highway network
        c = highway_network(c_char_embed, c_word_embed)
        q = highway_network(q_char_embed, q_word_embed)

        # contexual embedding layer
        c = self.context_lstm(c)[0]  # (batch_sz, c_len, 2*hidden_sz)
        q = self.context_lstm(q)[0]  # (batch_sz, q_len, 2*hidden_sz)

        # attention flow layer
        g = attn_flow_layer(c, q)    # (batch_sz, c_len, 8*hidden_sz)

        # modeling lstm layer
        m = self.model_lstm(g)[0]    # (batch_sz, c_len, 2*hidden_sz)
        p1, p2 = output_layer(g, m)  # (batch_sz, c_len), (batch_sz, c_len)
        return p1, p2
