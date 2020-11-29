[TOC]

## Transformer

### 1. Self-Attention

使用self-attention的原因：
+ 大多数Seq2Seq模型采用的结构均为Encoder-Decoder结构，并且，会结合Decoder中RNN的输出与Encoder中的RNN输出进行attention计算。但是使用RNN模型的弊端是它不能并行计算，因此也有部分学者提出用Text-CNN替代RNN。
+ CNN模型虽然解决了RNN模型不能并行计算的缺陷，但卷积核大小有限，于是它带来了另一个问题，模型无法获得长文本上下文的关联信息。因此，self-attention的设想由此而来。
+ 计算公式：$Attention(Q,K,T)=softmax(\frac{QK^T}{\sqrt{d}})V$

### 2. Multi-head self-attention
Multi-head attention是通过h个不同的线性变换对Q，K，V进行投影，最后将不同的attention结果拼接起来：
$$MultiHead(Q,K,V)=Concat(head_1, ..., head_h)W^O$$

### 3 Positional Encoding
因为Transformer抛弃了RNN，而RNN最大的优点就是在时间序列上对数据的抽象，所以文章中作者提出两种Positional Encoding的方法，将encoding后的数据与embedding数据求和，加入了相对位置信息。
作者选择使用不同频率的sine和cosine函数直接计算，公式如下：
$$PE_{pos,2i}=sin(pos/10000^{2i/d}$$
$$PE_{pos,2i+1}=cos(pos/10000^{2i/d}$$
采用此方式有两个好处：
+ 根据三角函数的特性，任意位置的$PE_{pos+k}$都可以被$PE_{pos}$的线性函数表示
+ 如果是学习到的positional embedding，会像词向量一样受限于词典大小。所以用三角公式则不受序列长度的限制，也就是可以对比所遇到序列的更长的序列进行表示。

### 4. 总结
Transformer是第一个用纯attention搭建的模型，不仅计算速度更快，在翻译任务上获得了更好的结果，也为后续的BERT模型做了铺垫。在BERT推出后，也涌现了越来越多的研究成果。


## 下周学习计划
+ 复习Bert模型结构
+ 预习几种Bert变体模型，如XLNet, Ernie
