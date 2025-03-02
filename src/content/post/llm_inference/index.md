---
title: "LLM Inference"
description: "llm inference notes"
publishDate: "13 July 2024"
updatedDate: "1 Dec 2024"
tags: ["tech/llm/optimization"]
draft: false
---

> 由于部署和调用LLM模型需求急速增加，迅速催生了LLM推理这一领域，围绕如何加快推理速度和成本首先从学术界出现大量结合系统领域知识的工作。本文是学习LLM推理的一些笔记。

## KV Cache
KV Cache是LLM推理优化中出现的第一个优化方法。理解KV Cache首先要了解LLM的推理过程的两点重要属性：1）**自回归(Autoregressive)**和2）Causal Masking。

自回归预测即next_token会加入到之前模型的输出中，再进行下一轮的预测。代码更能说明这个过程，见注释：
```python
@torch.no_grad()
def generate(self, input_idx, max_new_tokens, temperature=1.0):
    "Take a input sequence of indices and complete the sequence."
    for _ in range(max_new_tokens):
        idx_cond = input_idx if input_idx.size(1) <= self.block_size else input_idx[:, :self.block_size]
        # model接受用户输入进行计算，也称为profile阶段
        logits, _ = self(idx_cond) 
        # 注意logits只取最后一个单词，即预测next_token
        logits = logits[:, -1, :] / temperature # (b, c)
        prob = F.(logits, dim=-1)
        # idx_next = F.argmax(prob, dim=-1)
        idx_next = torch.multinomial(prob, num_samples=1) # (b, 1)
        # 这一步是Autogressive，输出token会加入到in_tokens
        # 模型会重新对加长的in_tokens进行推理
        input_idx = torch.cat((idx_cond, idx_next), dim = 1)

    return input_idx
```

`model(in_tokens)`这一步中的`in_tokens`会逐渐增加并重新进入`model`中经过Transformer中每一层进行计算，也就是说推理计算复杂度会随着`in_tokens`线性增加！

由于`in_tokens`之前的输出都是计算过的，是不是能在这里做优化呢？答案是可以，这推理过程中Attention计算中的[Causal Masking](https://www.s7ev3n.space/posts/transformer/#decoder)有关，即使用一个上三角矩阵来遮盖掉未来的信息。这和`in_tokens`的优化可以用下面例子来说明下。

假设`in_tokens`中有目前已经有两个`token`，即`in_tokens = [t1, t2]`，进入Transformer的每一层的MHA中每个head时，会将`t1`通过线性层映射成`q1, k1, v1`，然后计算注意力，为了说明问题只保留`q,k,v`的计算：
```
[q1*k1.T q1*k2.T] [1 0] [v1] = (q1 * k1.T) * v1
[q2*k1.T q2*k2.T] [1 1] [v2] = (q2 * k1.T) * v1 + (q2 * k2.T) * v2
```
然后使用`(q2*k1.T)*v1+(q2*k2.T)*v2`去预测下一个`token`，称之为`t3`。现在`in_tokens=[t1, t2, t3]`，输入到模型中再次进行计算：
```
[q1*k1.T q1*k2.T q1*k3.T] [1 0 0] [v1] = (q1 * k1.T) * v1
[q2*k1.T q2*k2.T q2*k3.T] [1 1 0] [v2] = (q2 * k1.T) * v1 + (q2 * k2.T) * v2
[q3*k1.T q3*k2.T q3*k3.T] [1 1 1] [v3] = (q3 * k1.T) * v1 + (q3 * k2.T) * v2 + (q3 * k3.T) * v3
```
我们看到，即使有`q1,q2`与`k3`的计算，但是由于causal masking，其值都无效，并且最后的输出`(q3*k1.T)*v1 +(q3*k2.T)*v2+(q3* k3.T)*v3`其实只与上一轮保存的`k1, v1, k2, v2`和当前这轮的`q3, k3, v3`即通过线性层映射后的结果有关。于是，我们缓存上一轮的`key`和`value`，这就是**KV Cache**！

:::tip
为什么不缓存`query`呢？看了上面的推导，应该很容易知道为什么不需要缓存上一轮的query！ ：）
:::

KV Cache的实现是在每一层Transformer层中的Attention部分，和上一层的`past_key_value`(即KV Cache)直接拼接，这篇博客[^1]的实现比较清晰：
```python
def mha(x, c_attn, c_proj, n_head, past_key_value=None):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # qkv projection
    # n_seq = 1 when we pass past_key_value, so we will compute new_q, new_k and new_v
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # split into qkv
    qkv = np.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> [3, n_seq, n_embd]

    if past_key_value:
        # qkv
        new_q, new_k, new_v = qkv  # new_q, new_k, new_v = [1, n_embd]
        old_k, old_v = past_key_value
        k = np.vstack([old_k, new_k]) # k = [n_seq, n_embd], where n_seq = prev_n_seq + 1
        v = np.vstack([old_v, new_v]) # v = [n_seq, n_embd], where n_seq = prev_n_seq + 1
        qkv = [new_q, k, v]

    current_cache = [qkv[1], qkv[2]]

    # split into heads
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))  # [3, n_seq, n_embd] -> [n_head, 3, n_seq, n_embd/n_head]

    # causal mask to hide future inputs from being attended to
    if past_key_value:
        causal_mask = np.zeros((1, k.shape[0]))
    else:
        causal_mask = (1 - np.tri(x.shape[0])) * -1e10  # [n_seq, n_seq]

    # perform attention over each head
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]  # [n_head, 3, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head]
    # merge heads
    x = np.hstack(out_heads)  # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]
    # out projection
    x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x, current_cache

```
[^1]: [Speeding up the GPT - KV cache](https://dipkumar.dev/becoming-the-unbeatable/posts/gpt-past_key_value/)

:::important
随着输入`in_tokens`变成超级长（1 Million），KV Cache也会成为显存杀手，其大小可能会远超模型权重，也成为后续工作优化的目标。
:::

## MHA->MQA->GQA->MLA 
在[Multi-Head Attention](https://www.s7ev3n.space/posts/transformer/#multi-head-attention)中，输入序列Embedding的`d_model`会被切分成`n_head`组，然后分别经过注意力计算后再`concat`起来还原`d_model`的长度。前面KV Cache最后提到过，当输入序列非常长，KV Cache会成为显存杀手，成为优化的目标，下面的MQA(Multi-Query Attention)，`Grouped-Query Attetion`和`Multi-head Latent Attention`都是对MHA的改进！

### Multi-Query Attention
MQA来自于论文[Fast Transformer Decoding: One Write-Head is All
You Need](https://arxiv.org/pdf/1911.02150)，来自于Transformer论文的第二作者Noam Shazeer。

想法很简单：只对`Query`进行切分成`n_head`组，形状变为`(b, n_heads, t, d_h)`，但是`Key`和`Value`直接通过线性层映射到形状`(b, t, d_h)`，如此以来`W_k`和`W_v`的参数量急剧减少！注意，此时`Key`和`Value`和`Query`的形状是不能直接矩阵相乘的，可以利用广播Boardcast原则，在`Key`和`Value`的第二个维度增加1，即`(b, 1, t, d_h)`，这样会在矩阵乘法的时候自动在`n_haeds`的维度扩充进行相乘。

但是，MQA有明显的缺点：性能下降严重！需要完全重新训练MQA的模型，才能带来推理速度的加快。模型训练异常昂贵，训练性能下降的MQA不太划算。

:::important
如果搜索引擎搜索MQA，可能会见到mean pool和uptrain等奇怪的术语，这些并不出现在MQA的原论文中，而是在下面章节的[GQA](https://arxiv.org/pdf/2305.13245)论文中的uptraining章节。

uptraining将已经训练好的MHA的模型更好的转换成MQA的模型，而不用重新训练。具体操作分为两部分：1）将使用MHA训练好的checkpoint中的所有头的`W_k`和`W_v`投影矩阵进行平均值，从而获得类似MQA原文中的、单个的`W_k`和`W_v`，即所谓的mean pool；2）对于合并后的模型，再使用训练非常小的轮次，让模型适应合并的权重，相当于训练得到了新的、省KV Cache的部署模型。实验发现这样比MQA的性能更好，而且更省计算。
:::

### Grouped Query Attention
[GQA](https://arxiv.org/pdf/2305.13245)是MHA和MQA的一般情况，其想法也很直接：如果一组`Key`和`Value`性能下降，那么多搞几组`Key`和`Value`吧。

### Multi-head Latent Attention
MLA出现在[Deepseek-V2](https://arxiv.org/abs/2405.04434)中，实现了比MHA性能好，并且KV Cache大幅降低！

[^2]: [缓存与效果的极限拉扯：从MHA、MQA、GQA到MLA](https://spaces.ac.cn/archives/10091)

## Inference Arithmetic
[分析transformer模型的参数量、计算量、中间激活、KV cache](https://zhuanlan.zhihu.com/p/624740065)
[Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)