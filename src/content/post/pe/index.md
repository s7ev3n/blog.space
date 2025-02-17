---
title: "Position encoding notes"
description: "various position encoding"
publishDate: "3 Aug 2024"
updatedDate: "4 Jan 2025"
tags: ["tech/transformer"]
draft: false
---
> Position Encoding是Transformer重要组成部分，通过对输入序列的位置进行编码，使模型有能力分辨不同位置的能力，主要有**绝对位置编码**和**相对位置编码**。本文是对苏神的[Transformer升级之路系列](https://spaces.ac.cn/archives/8231)[^1][^2]的个人笔记以及RoPE编码的学习。

[^1]: [Transformer升级之路1](https://spaces.ac.cn/archives/8231)
[^2]: [Transformer升级之路2](https://kexue.fm/archives/8265)
[^3]: [让研究人员绞尽脑汁的Transformer位置编码](https://kexue.fm/archives/8130)
[^4]: [层次分解位置编码，让BERT可以处理超长文本](https://kexue.fm/archives/7947)

## Absolute and Relative PE
### 绝对位置编码
一般来说，绝对位置编码会加到输入序列中：在输入的第$k$个向量$x_k$中加入位置向量$p_k$，即$x_k+p_k$，其中$p_k$只依赖于在序列的位置$k$。
常见的位置向量形式是三角式(Sinusoidal)和可训练式，也有其他形式[^3]。

相对于相对位置编码，绝对位置编码的优点是计算复杂度更低。

#### 三角函数(Sinusoidal)
三角函数(Sinusoidal)是Transformer论文中默认的位置编码，回顾一下：
$$
\begin{equation}\left\{\begin{aligned}&\boldsymbol{p}_{k,2i}=\sin\Big(k/10000^{2i/d}\Big)\\ 
&\boldsymbol{p}_{k, 2i+1}=\cos\Big(k/10000^{2i/d}\Big) 
\end{aligned}\right.\end{equation}
$$
其中，$p_{k,2i},p_{k,2i+1}$分别是序列中位置$k$处的、在`embed_dim`上的$2i,2i+1$的位置，$d$是`embed_dim`的大小。

在[Transformer](https://www.s7ev3n.space/posts/transformer/)文中，我们知道Sinusocidal位置编码在`embed_sim`后面接近于$0$和$1$间隔的编码，因此可以期望它有一定的**外推性**。

三角函数还有一个有意思的性质：$\sin(i+j)=\sin i\cos j+\cos i\sin j$和$\cos(i+j)=\cos i \cos j - \sin i \sin j$，即位置$i+j$可以表示成$i$和$j$的组合的形式，这提供了某种表达相对位置编码的性质。

:::important
**外推性**是指模型在推理阶段输入比训练阶段更长序列时的泛化能力。举例来说，预训练时的最大长度是$512$，但是在推理时输入了$768$长度的序列，由于位置编码在训练时没有见过这样长的序列，位置编码是否还可以提供有效的位置信息。
:::

#### 可训练式
可训练式位置编码不去设计编码的形式，而是将编码作为可学习的参数，与输入向量相加。在视觉任务的Transformer工作中，例如DETR及其后续工作，都是位置编码都是可训练式。

不难想象，可训练式位置编码的缺点是没有外推性，即推理时无法处理超过训练时最长长度的输入序列。不过，苏神的文章[^4]通过层次分解的方式，使得绝对位置编码能外推到足够的长度。

### 相对位置编码
相对位置并没有完整建模每个输入的位置信息，而是在计算attention的时候考虑当前位置与其他位置的相对距离，由于自然语言一般更依赖于相对位置，所以**相对位置编码通常有着优秀的表现**。对于相对位置编码来说，它的灵活性更大。但是，由于相对位置编码对Attention的计算进行了修改，它的计算复杂度和attention计算同样是$O(n^2)$，效率上显然低于绝对位置编码。另外，还是由于修改了Attention计算，后面对Attention的优化工作就无法执行。总的来说，相对和绝对位置编码是一个trade-off，而后面将要介绍的RoPE编码是融合了相对位置和绝对位置的一种编码方式，成为LLM的标配。

考虑一般形式的相对位置编码[^5]:
$$
\begin{equation}\left\{\begin{aligned} 
\boldsymbol{q}_i =&\, (\boldsymbol{x}_i + \boldsymbol{p}_i)\boldsymbol{W}_Q \\ 
\boldsymbol{k}_j =&\, (\boldsymbol{x}_j + \boldsymbol{p}_j)\boldsymbol{W}_K \\ 
\boldsymbol{v}_j =&\, (\boldsymbol{x}_j + \boldsymbol{p}_j)\boldsymbol{W}_V \\ 
a_{i,j} =&\, softmax\left(\boldsymbol{q}_i \boldsymbol{k}_j^{\top}\right)\\ 
\boldsymbol{o}_i =&\, \sum_j a_{i,j}\boldsymbol{v}_j 
\end{aligned}\right.\end{equation}
$$
其中$i$和$j$对应序列中的不同位置。

我们将$q_i$和$k_j$代入到$softmax$的公式的$q_i k_j^\top$中去，得到：
$$
\begin{equation} 
\boldsymbol{q}_i \boldsymbol{k}_j^{\top} = \left(\boldsymbol{x}_i + \boldsymbol{p}_i\right)\boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\left(\boldsymbol{x}_j + \boldsymbol{p}_j\right)^{\top} = \left(\boldsymbol{x}_i \boldsymbol{W}_Q + \boldsymbol{p}_i \boldsymbol{W}_Q\right)\left(\boldsymbol{W}_K^{\top}\boldsymbol{x}_j^{\top} + \boldsymbol{W}_K^{\top}\boldsymbol{p}_j^{\top}\right) 
\end{equation}
$$
作为对比，假如我们没有假如相对位置编码的偏置，应该是：
$$
\boldsymbol{q}_i \boldsymbol{k}_j^{\top}=\boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top} \boldsymbol{x}_j^{\top}
$$
那么，去掉$\boldsymbol{p}_i \boldsymbol{W}_Q$，并且将$\boldsymbol{p}_j \boldsymbol{W}_K$替换成$\boldsymbol{R}_{i,j}^{K}$:
$$
\begin{equation} 
a_{i,j} = softmax\left(\boldsymbol{x}_i \boldsymbol{W}_Q\left(\boldsymbol{x}_j\boldsymbol{W}_K + \color{green}{\boldsymbol{R}_{i,j}^K}\right)^{\top}\right) 
\end{equation}
$$
最后，在使用$v_i$计算加权和时:$\boldsymbol{o}_i =\sum\limits_j a_{i,j}\boldsymbol{v}_j = \sum\limits_j a_{i,j}(\boldsymbol{x}_j\boldsymbol{W}_V + \boldsymbol{p}_j\boldsymbol{W}_V)$，将$\boldsymbol{p}_j\boldsymbol{W}_V$替换成$\boldsymbol{R}_{i,j}^{V}$:
$$
\begin{equation}
\boldsymbol{o}_i = \sum_j a_{i,j}\left(\boldsymbol{x}_j\boldsymbol{W}_V + \color{green}{\boldsymbol{R}_{i,j}^{V}}\right) 
\end{equation}
$$
那么，$\boldsymbol{R}_{i,j}^{K}$和$\boldsymbol{R}_{i,j}^{V}$是什么？它们怎么体现出相对的位置关系的？
所谓相对位置，是"将本来依赖于二元坐标$(i,j)$的向量$\boldsymbol{R}_{i,j}^{K}, \boldsymbol{R}_{i,j}^{V}$，改为只依赖于相对距离$i−j$，并且通常来说会进行截断，以适应不同任意的距离":
$$
\begin{equation}\begin{aligned} 
\boldsymbol{R}_{i,j}^{K} = \boldsymbol{p}_K\left[\text{clip}(i-j, p_{\min}, p_{\max})\right] \\ 
\boldsymbol{R}_{i,j}^{V} = \boldsymbol{p}_V\left[\text{clip}(i-j, p_{\min}, p_{\max})\right]
\end{aligned}\end{equation}
$$
$\boldsymbol{p}_K$和$\boldsymbol{p}_V$是**可以是可训练式活三角函数式**的，都可以达到处理任意长度文本的需求。

相对位置编码还有一些形式，例如XLNET，T5或DeBERTa，都是对上面的一般式进行了一些变化[^3]。

[^5]: [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155)
## 旋转式位置编码
"一般来说，绝对位置编码具有实现简单、计算速度快等优点，而相对位置编码则直接地体现了相对位置信号，跟我们的直观理解吻合，实际性能往往也更好。由此可见，如果可以通过绝对位置编码的方式实现相对位置编码，那么就是集各家之所长。"

旋转式位置编码，英文是Rotary Position Embeddin (RoPE) 是一种“绝对位置编码的方式实现相对位置编码”的设计[^2]。