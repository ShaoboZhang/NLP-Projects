import numpy as np
import data_io as pio
from nltk import word_tokenize


class Preprocessor:
    def __init__(self, datasets_fp, max_length=384, stride=128):
        self.datasets_fp = datasets_fp
        self.max_length = max_length
        self.max_clen = 100
        self.max_qlen = 100
        self.stride = stride
        self.charset = set()
        self.ch2id, self.id2ch = {}, {}
        self.build_charset()

    def build_charset(self):
        for fp in self.datasets_fp:
            self.charset |= self.dataset_info(fp)
        self.charset = sorted(list(self.charset))
        self.charset = ['[PAD]', '[CLS]', '[SEP]'] + self.charset + ['[UNK]']
        idx = list(range(len(self.charset)))
        self.ch2id = dict(zip(self.charset, idx))
        self.id2ch = dict(zip(idx, self.charset))

    def dataset_info(self, inn):
        charset = set()
        dataset = pio.load(inn)
        for _, context, question, answer, _ in self.iter_cqa(dataset):
            charset |= set(context) | set(question) | set(answer)
            # self.max_clen = max(self.max_clen, len(context))
            # self.max_qlen = max(self.max_clen, len(question))
        return charset

    @staticmethod
    def iter_cqa(dataset):
        for data in dataset['data']:
            for paragraph in data['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    qid = qa['id']
                    question = qa['question']
                    for answer in qa['answers']:
                        text = answer['text']
                        answer_start = answer['answer_start']
                        yield qid, context, question, text, answer_start

    def encode(self, context, question):
        question_encode = self.convert2id(question, begin=True, end=True)
        left_length = self.max_length - len(question_encode)
        context_encode = self.convert2id(context, maxlen=left_length, end=True)
        cq_encode = question_encode + context_encode
        assert len(cq_encode) == self.max_length
        return cq_encode

    def convert2id(self, sent, maxlen=None, begin=False, end=False):
        ch = [ch for ch in sent]
        ch = ['[CLS]'] * begin + ch
        if maxlen is not None:
            ch = ch[:maxlen - 1 * end]
            ch += ['[SEP]'] * end
            ch += ['[PAD]'] * (maxlen - len(ch))
        else:
            ch += ['[SEP]'] * end
        ids = list(map(self.get_id, ch))
        return ids

    def get_id(self, ch):
        return self.ch2id.get(ch, self.ch2id['[UNK]'])

    def get_dataset(self, ds_fp):
        cs, qs, be = [], [], []
        cws, qws = [], []
        for _, c, q, c_w, q_w, b, e in self.get_data(ds_fp):
            cs.append(c)
            qs.append(q)
            cws.append(c_w)
            qws.append(q_w)
            be.append((b, e))
        return map(np.array, (cs, qs, cws, qws, be))

    def get_data(self, ds_fp):
        dataset = pio.load(ds_fp)
        for qid, context, question, text, answer_start in self.iter_cqa(dataset):
            cids = self.get_sent_ids(context, self.max_clen)
            qids = self.get_sent_ids(question, self.max_qlen)
            c_words = self.tokenize(context)
            q_wrods = self.tokenize(question)
            b, e = answer_start, answer_start + len(text)
            if e >= len(cids):
                b = e = 0
            yield qid, cids, qids, c_words, q_wrods, b, e

    def get_sent_ids(self, sent, maxlen):
        return self.convert2id(sent, maxlen=maxlen, end=True)

    # 利用nltk进行分词，实现word级别编码
    def tokenize(self, sent):
        words = word_tokenize(sent)
        words = ['[CLS]'] + words[:self.max_length - 1]
        words += ['[PAD]'] * (self.max_length - len(words))
        return words


# 加载Glove到Embedding matrix
def load_glove(vocab_size=20000, fp='./data/glove/glove.6B.50d.txt'):
    """
    从glove文件中加载词向量，并生成词向量矩阵
    :param vocab_size: 词汇表最大单词数量
    :param fp: glove文件的存储路径
    :return word2idx: 单词到索引的字典
    :return embed: 生成的词向量矩阵
    """
    # 保存glove中的单词
    vocab = []
    # 保存glove中单词对应的词向量
    matrix = []
    with open(fp, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f.readlines()):
            line = line.strip().split()
            vocab.append(line[0])
            matrix.append(line[1:])
            if idx +1 == vocab_size:
                break
    # print("glove词向量加载完成")
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    return word2idx, np.array(matrix)


if __name__ == '__main__':
    p = Preprocessor([
        './data/squad/train-v1.1.json',
        './data/squad/dev-v1.1.json'
    ])
    # print(p.encode('modern stone statue of Mary', 'To whom did the Virgin Mary'))
    # word2idx, embed = load_glove()
    batch = p.get_data('./data/squad/dev-v1.1.json')
    pass