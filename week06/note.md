## 1 BERT的问题
BERT发布后，在排行榜上产生了许多NLP任务的最新成果。但是，模型非常大，导致了一些问题。
+ 内存限制：BERT-large是一个复杂的模型，它有24个隐藏层，在前馈网络和多头注意力机制中有很多节点，总共有3.4亿个参数，如果想要从零开始训练，需要花费大量的计算资源。
+ 模型退化：最近在NLP领域的研究趋势是使用越来越大的模型，以获得更好的性能。而ALBERT的研究表明，无脑堆叠模型参数可能导致效果降低。

因此，在bert模型提出后，不断有新的模型出现，在Bert基础上进行改进，得到新的模型。

## 2 ALBERT
ALBERT利用了参数共享、矩阵分解等技术大大减少了模型参数，用SOP（Sentence Order Prediction）任务取代NSP（Next Sentence Prediction）任务，提升了下游任务的表现。但是ALBERT的层数并未减少，因此推理时间（Inference Time）还是没有得到改进。不过参数减少的确使得训练变快，同时ALBERT可以扩展到比BERT更大的模型（ALBERT-xxlarge），因此能得到更好的表现。

ALBERT的结构和BERT基本一样，采用了Transformer以及GELU激活函数。具体的创新部分有三个：
1. embedding 层参数因式分解
2. 跨层参数共享
3. 将NSP任务改为SOP任务

其中，前两个改进的主要作用是减少参数。


## 3 RoBERTa
RoBERTa相较于BERT最大的改进有三点：
1. 动态Masking
2. 取消NSP (Next Sentence predict) 任务
3. 扩大Batch Size


## 4 StructBERT
StructBERT是阿里对的一个BERT改进，模型取得了很好的效果，目前在GLUE排行榜排名靠前。

对于一个人来说，字或字符的顺序不影响阅读，模型也是一样，一个好的LM，需要懂得自己纠错，这就是structBERT的改进思路的来源。

StructBERT的模型架构和BERT一样，它改进在于，在现有MLM和NSP任务的情况下，新增了两个预训练目标：Word Structural Objective 和Sentence Structural Objective。