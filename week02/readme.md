[TOC]

## Bi-DAF
本次工程的目的是实现Bi-DAF模型，包括如下几个文件夹与文件
+ data: 存放工程中所用的数据集和词向量文件，包括：
    1. glove文件夹: 存放glove词向量文件
    2. squad文件夹: 存放此次模型用到的SQuAD数据集
+ layers: 模型中各层的类函数，包括：
    1. highway.py: Highway Network，得到的数据经过char embedding+CNN与word embedding并拼接后输入的网络
    2. similarity.py: 用以计算context word与query word之间的相似度矩阵
    3. attention.py: 注意力层，context与query经过Highway Network后进行交互计算注意力的网络
    4. merge.py: 经过注意力计算之后，对解码的输入文本进行按相似度进行加权
    5. span.py: 输出层，用以预测开始位置p1和结束位置p2
+ data_io.py: 存放关于读取数据的辅助函数
+ preprocess.py: 用以对数据进行预处理操作，生成方便后续模型调用的数据格式
+ main.py: 主程序，用以搭建模型并进行训练