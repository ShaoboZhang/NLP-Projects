[TOC]

## BiDAF中的attention
为与论文保持一致，T表示上下文文本长度，J表示问题文本长度。用H表示经过LSTM之后的上下文编码，维度为2d*T，U表示问题编码，维度为2d*J。

### 1. Similarity Matrix
在BiDAF中，计算attention的第一步，就是计算相似矩阵(similarity matrix)S，维度为T*J，其中每i行表示的是上下文文本中第i个词与问题文本中每一个词之间的相似度，第j列表示的是问题中第j个词与上下文文本中每一个词的相似度。

而矩阵中每个位置相似度的计算方法由 $S_{ij}=W_{(s)}^T*[h;u;h.*u]$ 得到。该公式的含义是H的第i列与U的第j列按位点乘后再与他们自己拼接到一起，形成一个维度为6d的向量，然后与$W_{(s)}$相乘得到相似度分数$S_{ij}$。

### 2. Context-to-Query Attention
C2Q注意力计算的是对context中每个单词，query中哪个单词与其关联性最大，其计算公式如下：
$$a_t=softmax(S_{t:})\in R^J$$
$$\hat{U}_{:t}=\sum_ja_{tj}U_{:j}$$
具体讲，就是将S相似度矩阵每一行经过softmax层直接作为注意力值，因为S中每一行表示的是上下文文本中第i个词与问题文本中每一个词之间的相似度，C2Q表示的是文本对问题的影响，所以得到$a_t$就直接与U中的每一列加权求和得到新的$\hat{U}_{:t}$，最后拼成新的问题编码，它是一个2d*T的矩阵。

### 3. Query-to-Context Attention(C2Q)
计算的是对每一个query单词而言，哪些context单词和它最相关，因为这些context单词对回答问题很重要，故直接取相关性矩阵中最大的那一列，再进行softmax归一化计算context向量加权和，然后重复T次得到${\hat{H}\in R^{2d*T}}$

### 4. 代码实现
在代码实现过程中，为了避免使用for loop计算每个位置，可以对变量进行复制及维度扩展，而后尽可能利用矩阵乘法的得到相似度矩阵。
```python

    self.attn_c = nn.Linear(args.hidden_size * 2, 1)
    self.attn_q = nn.Linear(args.hidden_size * 2, 1)
    self.attn_cq = nn.Linear(args.hidden_size * 2, 1)

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
```


## QANet

在QANet模型中，该模型架构与后续的Bert模型有很多相似之处，这里着重谈谈其中的FeedForward Layer和LayerNorm Layer。

### 1. FeedForward Network
Feedforward Networks 也叫Multilayer Perceptrons(多层感知机)，将上一层传过来的输入信息做线性变换，再经过一个激活函数输出给下一层。其公式为：
$$Y=\sigma(W^TX)$$
公式中的$\sigma$是激活函数，通常为ReLU函数或tanh函数。因为单纯的矩阵乘法其本质还是线性变换，所以引入激活函数希望对输入进行非线性变换。

### 2. Layer Normalization
通常，在经过Embedding层之后，数据的维度一般为$(batch size, seq len, hidden size)$。
对于NLP任务而言，同一batch下不同数据之间没有关联性，而不同数据间，每句话的长度也不一样。因此在第0维和第1维做normalization的意义不大，所以是在数据的第2维，也即特征维度进行归一化。

## 下周学习计划
+ 学习理解Transformer模型的Encoder层，这也是Bert模型的来源
+ 学习理解Bert模型