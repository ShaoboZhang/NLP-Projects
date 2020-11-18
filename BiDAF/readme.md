## BiDAF

此项目旨在构建BiDAF模型，利用SQUAD数据集验证模型效果。
模型使用pytorch框架，各目录内容包括：
+ data: 存放实验所用的数据，子目录包括glove和sqaud
    + glove: 存放生成Glove词向量的文档
    + squad: 存放SQUAD1.0数据集
+ model: 存放训练后的模型参数
+ utils: 存放此次工程所用的辅助函数文件，包括：
    + config.py: 模型训练时用到的各项参数，例如文件路径、句子长度、词表大小、词向量维度等
    + preprocess.py: 用于文本预处理，生成词向量矩阵及模型的输入
    + model.py: 构建BiDAF模型
+ main.py: 主程序文件