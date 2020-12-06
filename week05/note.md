[TOC]

## XLNet
XLNet论文作者表示，BERT这样基于去噪自编码器的预训练模型可以很好地建模双向语境信息，性能优于基于自回归语言模型的预训练方法。然而，由于需要mask一部分输入，BERT忽略了被mask位置之间的依赖关系，因此出现预训练和微调效果的差异（pretrain-finetune discrepancy）

基于这些优缺点，该研究提出了一种泛化的自回归预训练模型 XLNet。XLNet 可以：1）通过最大化所有可能的因式分解顺序的对数似然，学习双向语境信息；2）用自回归本身的特点克服 BERT 的缺点。此外，XLNet 还融合了当前最优自回归模型 Transformer-XL 的思路

以前超越 BERT 的模型很多都在它的基础上做一些修改，本质上模型架构和任务都没有太大变化。但是在这篇新论文中，作者从自回归（autoregressive）和自编码（autoencoding）两大范式分析了当前的预训练语言模型，并发现它们虽然各自都有优势，但也都有难以解决的困难。为此，研究者提出 XLNet，并希望结合大阵营的优秀属性。

最终，XLNet 在 20 个任务上超过了 BERT 的表现，并在 18 个任务上取得了当前最佳效果（state-of-the-art），包括机器问答、自然语言推断、情感分析和文档排序。

## AR与AE
### 自回归语言模型（AutoRegressive LM）
通常讲的语言模型其实是根据上文内容预测下一个可能跟随的单词，就是常说的自左向右的语言模型任务，或者反过来也行（就是根据下文预测前面的单词）。这种类型的LM被称为自回归语言模型。

给定文本序列$x = [x_1,...,x_T]$，语言模型的目标是调整参数使得训练数据上的似然函数最大：
$$\max_{\theta}logp_\theta(x)=\sum^T_{t=1}logp_\theta(x_t|x_{<t})$$

其中，$x_{<t}$表示t时刻之前的所有x，也就是$x_{1:t-1}$。

### 自编码语言模型（AutoEncoder LM）
BERT通过将序列中随机挑选15%的Token变成[MASK]得到带噪声版本的。假设被Mask的原始值为，那么BERT希望尽量根据上下文恢复（猜测）出原始值，也就是：
$$\max_{\theta}logp_\theta(\overline{x}|\hat{x})\approx\sum^T_{t=1}m_tlogp_\theta(x_t|\hat{x})$$

其中，$m_t=1$表示t时刻是一个 Mask，需要恢复。

XLNet的出发点就是：能否融合AR和AE语言模型两者的优点。具体来说就是，站在AR的角度，如何引入和双向语言模型等价的效果。


## Permutation Language Model
具体的解决方式是：在AR以及AE方式中再加入一个步骤，就能够完美地将两者统一起来，那就是Permutation。

具体实现方式是，通过随机取一句话排列的一种，然后将末尾一定量的词给 “遮掩”（和BERT里的直接替换"[MASK]"有些不同）掉，最后用AR的方式来按照这种排列方式依此预测被“遮掩”掉的词。


## 完整版代码
xlnet完整版代码 https://github.com/ShaoboZhang/NLP-Projects/blob/main/week05/xlnet.py