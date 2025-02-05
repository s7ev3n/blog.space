---
title: "Position encoding notes"
description: "various position encoding"
publishDate: "3 Aug 2024"
updatedDate: "4 Feb 2025"
coverImage:
  src: ""
  alt: ""
tags: ["tech/transformer"]
draft: false
---
> Position Encoding是Transformer重要组成部分，用来模型对输入序列的位置信息进行编码。主要有**绝对位置编码**，将位置信息嵌入到输入序列中，以及**相对位置编码**，通过微调Attention结构，使模型有能力分辨不同位置的Token。本文是对苏神的[Transformer升级之路系列](https://spaces.ac.cn/archives/8231)[^1]的个人笔记，主要包括对Sinusoidal位置编码的更深入的理解，以及RoPE编码的学习。

[^1]: [Transformer升级之路1](https://spaces.ac.cn/archives/8231)
[^2]: [Transformer升级之路2](https://kexue.fm/archives/8265)
[^3]: [让研究人员绞尽脑汁的Transformer位置编码](https://kexue.fm/archives/8130)
[^4]: [层次分解位置编码，让BERT可以处理超长文本](https://kexue.fm/archives/7947)

## Absolute and Relative PE
### 绝对位置编码
一般来说，绝对位置编码会加到输入序列中：在输入的第$k$个向量$x_k$中加入位置向量$p_k$，即$x_k+p_k$，其中$p_k$只依赖于在序列的位置$k$。

常见的位置向量形式是三角函数和可学习参数，也有其他形式[^3]。

### 外推性


### 相对位置编码
相对位置并没有完整建模每个输入的位置信息，而是在算Attention的时候考虑当前位置与被Attention的位置的相对距离，由于自然语言一般更依赖于相对位置，所以相对位置编码通常也有着优秀的表现。对于相对位置编码来说，它的灵活性更大。

$$
\begin{equation}\left\{\begin{aligned} 
\boldsymbol{q}_i =&\, (\boldsymbol{x}_i + \boldsymbol{p}_i)\boldsymbol{W}_Q \\ 
\boldsymbol{k}_j =&\, (\boldsymbol{x}_j + \boldsymbol{p}_j)\boldsymbol{W}_K \\ 
\boldsymbol{v}_j =&\, (\boldsymbol{x}_j + \boldsymbol{p}_j)\boldsymbol{W}_V \\ 
a_{i,j} =&\, softmax\left(\boldsymbol{q}_i \boldsymbol{k}_j^{\top}\right)\\ 
\boldsymbol{o}_i =&\, \sum_j a_{i,j}\boldsymbol{v}_j 
\end{aligned}\right.\end{equation}
$$

## Sinusoidal位置编码


## 旋转式位置编码
旋转式位置编码，英文是Rotary Position Embeddin (RoPE)是LLM中标配的位置编码，是一种“绝对位置编码的方式实现相对位置编码”的设计[^2]。