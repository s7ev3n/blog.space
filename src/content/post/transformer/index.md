---
title: "Transformer notes"
description: "transfomer learning notes and code implementation"
publishDate: "18 Jan 2025"
updatedDate: "18 Jan 2025"
coverImage:
  src: "./figs/transformer_components.png"
  alt: "Transformer components"
tags: []
draft: false
---

# Transformer notes
Transformer主要由题图中的三个部分组成：scaled dot-product attention, multi-head attention，Transformer achitecture。笔记主要以这三部分为大纲，每个部分会包括模块的解读和代码实现细节。

<details>
<summary>感性理解注意力</summary>
听说过Transformer的人一定会见到Query, Key, Value这几个东西，为什么Query和Key要想相乘得到相似度后与Value进行加权和？ 如果你也有这样的疑问，可以从下面的内容有一个感性认识，如果只想了解技术部分，这里完全可以跳过。参考资料来自[动手深度学习中注意力机制](https://zh.d2l.ai/chapter_attention-mechanisms/index.html)

心理学中威廉·詹姆斯提出了双组件(two-component)框架：受试者基于**自主性提示**和**非自主性提示**有选择的引导注意力的焦点。自主性提示就是人主观的想要关注的提示，而非自主性提示是基于环境中物体的突出性和易见性。举一个下面的例子：

想象一下，假如你面前有五个物品： 一份报纸、一篇研究论文、一杯咖啡、一本笔记本和一本书，如下图。 所有纸制品都是黑白印刷的，但咖啡杯是红色的。 
这个咖啡杯在这种视觉环境中是突出和显眼的， 不由自主地引起人们的注意，属于**非自主性提示**。
但是，受试者可能更像看书，于是会主动、自主地去寻找书，选择认知和意识的控制，属于**自主性提示**。

**将上面的自主性提示、非自主性提示与“查询query、键key和值value”联系起来**
作为对比：查询query相当于自主性提示，键key相当于非自主性提示，而值value相当于提示对应的各种选择，因而键key和值value是成对出现的。下图框架构建了注意力机制：

<div style="text-align: center">
    <figure style="display: inline-block">
        <img src="./figs/qkv.svg" alt="qkv" width="400">
        <figcaption>Fig. attention</figcaption>
    </figure>
</div>

</details>

## 缩放点积注意力(scaled dot-product attention)
缩放点积注意力模块由注意力评分函数和加权求和组成。

注意力评分函数$f_{attn}$是$\mathbf{query}$向量和$\mathbf{key}$向量的点积，即向量之间的相似度，并除以向量的长度$d$($\mathbf{query}$和$\mathbf{key}$具有相同的长度$d$):

$$
f_{attn}(\mathbf q, \mathbf k) = \frac{\mathbf{q} \mathbf{k}^\top }{\sqrt{d}} \in \mathbb R^{b \times n \times m}
$$ 

$\mathbf{query}$，$\mathbf{key}$ 和 $\mathbf{value}$都是张量的形式，例如 $\mathbf q\in\mathbb R^{b \times n\times d}$，$\mathbf k\in\mathbb R^{b \times m\times d}$，$\mathbf v\in\mathbb R^{b \times m \times v}$，其中$b$代表batch，有$n$个查询$\mathbf{query}$，$m$个$\mathbf{key}$和$\mathbf{value}$。

你可能注意到了$\mathbf{query}$的数量$n$可以和$\mathbf{key}$的数量$m$不同，但是向量的长度$d$必须相同；$\mathbf{key}$和$\mathbf{value}$的数量必须相同，但是向量的长度可以不同。但是在Transformer的自注意力self-attention中，由于是自注意力，数量和向量长度都是相同的。

最后，缩放点积注意力模块是对$\mathbf{value}$的加权和：

$$
\mathrm{softmax}\left(\frac{\mathbf q \mathbf k^\top }{\sqrt{d}}\right) \cdot \mathbf V \in \mathbb{R}^{b \times n\times v}
$$ 

图中还有mask的部分，将会在[后面](#masked-multi-head-attention)进行说明。

### 实现scaled dot-product attention
用最原始的代码实现一下Transformer中的attention：

```python
def attention(query, key, value, attn_mask=None, dropout=None):
  """Scaled Dot Product Attention.

    Attention(Q, K, V) = softmax(Q * K.T / sqrt(d_k)) * V
    
    Params:
        query: (b, t, d_k)
        key  : (b, t, d_k)
        value: (b, t, d_k)
    Returns:
        result: (b, t, d_k)
        attn  : (b, t, t)
        
    Attetion detail: 
    a query vector (1, d_k) calcuates its similarity (vector dot product)
    with a sequence key vectors (t, d_k), and the output (1, t) is the query's
    attention with t key vectors, by multiplying with value (t, d_k), the
    output (1, d_k) is a weighted sum over value features, which is the most
    representative features related with query feature. It could easily extend
    to a sequence of query vectors (t, d_k), the output is a attention matrix of 
    shape (t, t), the rest is the same.
    
    A more concrete example, suppose query (3, 2), key (3, 2) and value (3, 2),
    the attention matrix (3, 3) show below:
                               [1.0 , 0.0 , 0.0 ]
                               [0.5 , 0.5 , 0.0 ]
                               [0.33, 0.33, 0.33]
    Let's make value vector (3, 2) more concrete to see weighted sum over value
    (keep in mind that each row in value vector (1, 2) is a feature vector):
                                   [1, 2]
                                   [4, 5]
                                   [7, 8]
    and after attn * value:
                                 [1.0, 2.0]
                                 [2.5, 3.5]
                                 [4.0, 5.0]
    Each element in a row of attention matrix specifies how each value vector is 
    summed, 
    e.g. [0.5, 0.5, 0.0] specifies 0.5 * [1 2] + 0.5 * [4 5] + 0 * [7 8] = [2.5 3.5]
    PS: see video 
    https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7&t=2533s
    """
    d_k = query.size(-1)
    score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(d_k) # (b, t, t)

    if attn_mask is not None:
        # NOTE: Why set mask position to -np.inf ?
        # 1. Make sure masking position has no effect, set to 0 DO NOT lead to probability 0 using softmax!
        # 2. Softmax will give close to 0.0 prob to -np.inf but not 0.0 to avoid gradient vanishing
        # 3. For computation stability, to avoid underflow
        score = score.masked_fill(attn_mask == 0, -1e9)
    
    attn = nn.functional.softmax(score, dim=-1)
    if dropout is not None:
        # TODO: Why dropout here ?
        attn = dropout(attn)

    return torch.matmul(attn, value), attn

```

## 多头注意力(multi-head attention)

多头注意力将$\mathbf{query}$，$\mathbf{key}$和$\mathbf{value}$的向量长度$d$切分成更小的几($n\_heads$)组，每组称为一个头，每个头的向量长度是$d=\frac{d_{model}}{n\_heads}$，每个头内进行缩放点积注意力计算，并在每个头计算结束后连结(`concat`)起来，再经过一个全连接层后输出，如下图所示：

<div style="text-align: center">
    <figure style="display: inline-block">
        <img src="./figs/multi-head-attention.svg" alt="mha" width="400">
        <figcaption>Fig. multi-head attention</figcaption>
    </figure>
</div>


给定$\mathbf{q} \in \mathbb{R}^{d_q}$、$\mathbf{k} \in \mathbb{R}^{d_k}$和$\mathbf{v} \in \mathbb{R}^{d_v}$，每个注意力头$h_i(i=1,...,h)$的计算方法为：

$$
\mathbf{h}_i = f(\mathbf W_i^{(q)}\mathbf q, \mathbf W_i^{(k)}\mathbf k,\mathbf W_i^{(v)}\mathbf v) \in \mathbb R^{p_v}
$$

其中，可学习的参数包括$\mathbf W_i^{(q)}\in\mathbb R^{p_q\times d_q}$，$\mathbf W_i^{(k)}\in\mathbb R^{p_k\times d_k}$和$\mathbf W_i^{(v)}\in\mathbb R^{p_v\times d_v}$

多头注意力的输出需要经过另一个全连接层转换， 它对应着$h$个头连结(concat)后的结果，因此其可学习参数是$\mathbf W_o\in\mathbb R^{p_o\times h p_v}$:

$$
\begin{split}
\mathbf W_o \begin{bmatrix}\mathbf h_1\\\vdots\\\mathbf h_h\end{bmatrix} \in \mathbb{R}^{p_o}.
\end{split}
$$

其中$n\_heads$是超参数，存在：$p_q \cdot n\_heads = p_k \cdot n\_heads = p_v \cdot n\_heads = p_o$关系。

multi-head attention的实现：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.l = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask = None):
        # q, k, v : (b, t, d_model)
        b, t, d_model = q.size()

        q = self.W_q(q) # (b, t, d_model)
        k = self.W_k(k)
        v = self.W_v(v)
        
        q = q.view(b, t, self.n_heads, d_model // self.n_heads).transpose(1, 2)
        k = k.view(b, t, self.n_heads, d_model // self.n_heads).transpose(1, 2)
        v = v.view(b, t, self.n_heads, d_model // self.n_heads).transpose(1, 2) # (b, n_heads, t, d_k)
        
        x, attn = attention(q, k, v, attn_mask=mask, dropout=self.dropout)
        # x -> (b, n_heads, t, d_k), attn -> (b, n_heads, t, t)
        x = x.transpose(1, 2) # -> (b, t, n_heads, d_k)
        # it is necessary to add contiguous here
        x = x.contiguous().view(b, t, d_model) # -> (b, t, n_heads * d_k)
        res = self.l(x) # (b, t, d_model)
    
        return res 
```

## Transformer模型结构

`input embedding`在进入编码器Encoder前，通过与Positional Encoding相加获得位置信息，(<span style="color: gray">Positional Encoding只在这里输入相加一次，与DETR，DETR3D等视觉transformer不同</span>）。
编码器encoder有两部分：注意力multi-head attention模块和Feedforwad模块，每个某块都包括一个残差连接residual，并且这里有一个比较重要的细节是Norm的位置，图中所示是post-norm，而目前很多实现中使用的是pre-norm。

### 位置编码 positional encoding
可以注意到注意力机制是没有学习到位置信息的，即打乱 $n$ 个query向量的顺序，得到的注意力输出的值是没有变化的。因此，需要显式地给每个query向量提供位置信息。位置编码向量是与query向量维度相同的向量，位置变量向量通过公式得到，也可以学习得到，位置编码向量与query向量相加，可以将位置信息编码到query向量中，即打乱 $n$ 个query向量的顺序，会得到不同的注意力的值。

假设输入序列$\mathbf{X} \in \mathbb{R}^{n \times d}$ 是包含$n$个长度为$d$的query向量的矩阵，位置编码使用相同形状的位置嵌入矩阵$\mathbf{P} \in \mathbb{R}^{n \times d}$，并和输入相加得到输出$\mathbf{X} + \mathbf{P}$，矩阵第$i$行(表示序列中的位置)，第$2j$列和第$2j+1$列(表示每个位置的值)的元素为：

$$
\begin{aligned} 
p_{i, 2j} &= \sin\left(\frac{i}{10000^{2j/d}}\right) \\
p_{i, 2j+1} &= \cos\left(\frac{i}{10000^{2j/d}}\right)
\end{aligned}
$$

可以理解为在一列上是交替sin和cos的函数，并且沿着编码维度三角函数的频率单调降低。为什么频率会降低？以二进制编码类比下，看0-8的二进制表示：

```text
  0的二进制是：000
  1的二进制是：001
  2的二进制是：010
  3的二进制是：011
  4的二进制是：100
  5的二进制是：101
  6的二进制是：110
  7的二进制是：111
```
在二进制表示中，较高比特位的交替频率低于较低比特位。类比位置编码向量，一行中前面的元素的交替频率要高于后面的元素。

<details>
<summary>PositionEncoding的实现</summary>

```python
class PositionEncoding(nn.Module):
    """Position Encoding.

    Positional encoding will sum with input embedding to give input embedding order.
    Positional encoding is given by the following equation:
    
    PE(pos, 2i)     = sin(pos / (10000 ^ (2i / d_model)))
    PE(pos, 2i + 1) = cos(pos / (10000 ^ (2i / d_model)))
    # for given position odd end even index are alternating
    # where pos is position in sequence and i is index along d_model.
    
    The positional encoding implementation is a matrix of (max_len, d_model), 
    this matrix is not updated by SGD, it is implemented as a buffer of nn.Module which 
    is the state of of the nn.Module.
    
    Note: For max_len, it usually aligns with the sequence length, do not have to be 1024.

    Detail 1:
    In addition, we apply dropout to the sums of the embeddings and the positional encodings 
    in both the encoder and decoder stacks. For the base model, we use a rate of P_drop = 0.1
    """
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, requires_grad=False)
        pos = torch.arange(0, max_len).unsqueeze(1) # (max_len, 1)
        demonitor = torch.pow(10000, torch.arange(0, d_model, 2) / d_model) # pos/demonitor is broadcastable
        
        pe[:, 0::2] = torch.sin(pos / demonitor)
        pe[:, 1::2] = torch.cos(pos / demonitor)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (b, t, d_model)
        # self.pe[:, :x.size(1)] will return a new tensor, not buffer anymore
        # by default the new tensor's requires_grad is Fasle, but here we refer
        # to The Annotated Transformer, use in_place requires_grad_(False)
        x = x + self.pe[:, : x.size(1)].requires_grad_(False) # max_len is much longer than t
        return self.dropout(x)
```
</details>

### 编码器 encoder block
编码器encoder中第一个细节是`pre-norm`和`post-norm`：
```python
class SublayerResidual(nn.Module):
    def __init__(self, d_model=512, dropout=0.1):
        super(SublayerResidual, self).__init__()
        self.ln = nn.LayerNorm(d_model) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        x: input
        sublayer: MHA or FFN

        Detail 1:
        Note implementation here is pre-norm formulation:
            x + sublayer(LayerNorm(x))       
        Origin paper is using post-norm:
            LayerNorm(x+sublayer(x))
        There are literatures about the pros and cons of pre-norm and post-norm[1,2].

        Detail 2:
        We apply dropout to the output of each sub-layer, before it is added to the 
        sub-layer input and normalized.

        Reference: 
        1. https://youtu.be/kCc8FmEb1nY?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&t=5723
        2. https://kexue.fm/archives/9009
        """
        return x + self.dropout(sublayer(self.ln(x)))
```
第二细节也是问题：**为什么使用的是LayerNorm，而不是CNN时代的BatchNorm ?**

### 掩码自注意力 masked multi-head attention
掩码发生在 `attention` 函数中，将key和value相乘得到的attention score matrix，根据一个masked attention matrix遮盖掉不需要的attention score：
```python
if attn_mask is not None:
    score = score.masked_fill(attn_mask == 0, -1e9)
```

传入的masked attention matrix有很多名称，但是核心的目的只有一个，就是将未来的信息去掉
```python
def causal_masking(seq_len):
    """Masking of self-attention.

    The masking has many names: causal masking, look ahead masking, subsequent masking
    and decoder masking, etc. But the main purpose is the same, mask out after the 
    position i to prevent leaking of future information in the transformer decoder. 
    Usually, the mask is a triangular matrix where the elements above diagnal is True
    and below is False. 

    Args:
        seq_len (int): sequence length 
    """

    mask = torch.triu(torch.ones((1, seq_len,seq_len)), diagonal=1).type(torch.int8)
    
    return mask == 1
```

为什么要mask掉未来的信息？


### 整体结构