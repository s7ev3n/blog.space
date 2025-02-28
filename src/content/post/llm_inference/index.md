---
title: "LLM Inference"
description: "llm inference notes"
publishDate: "13 July 2024"
updatedDate: "1 Feb 2025"
tags: ["tech/llm/optimization"]
draft: false
---

> 由于部署和调用LLM模型需求急速增加，迅速催生了LLM推理这一领域，围绕如何加快推理速度和成本首先从学术界出现大量结合系统领域知识的工作。本文是学习LLM推理的一些笔记。

## KV Cache
KV Cache是LLM推理中最重要的概念之一。理解KV Cache首先要了解LLM的推理过程是自回归(Autoregressive)形式，即next_token会加入到之前模型的输出中，再进行下一轮的预测。

使用代码更能说明这个过程，见注释：
```python
tokenizer = GPT2Tokenizer.from_pretrained("gpt-2")
in_text = "Hello, my name is"
in_tokens = torch.tensor(tokenizer.encode(in_text))

# inference
token_eos = torch.tensor([198]) # 结束符
out_token = None
i = 0
with torch.no_grad():
    while out_token != token_eos:
        # model接受用户输入进行计算，也称为profile阶段
        logits, _ = model(in_tokens)
        # 注意logits只取最后一个单词，即预测next_token
        out_token = torch.argmax(logits[-1, :], dim=0, keepdim=True)
        # 这一步是Autogressive，输出token会加入到in_tokens
        # 模型会重新对加长的in_tokens进行推理
        in_tokens = torch.cat((in_tokens, out_token), 0)
        text = tokenizer.decode(in_tokens)
        print(f'step {i} input: {text}', flush=True)
        i += 1
```

`model(in_tokens)`这一步中的`in_tokens`会逐渐增加并重新进入`model`中经过Transformer中每一层进行计算，但是前一步的`in_tokens`全部是计算过的！只有新加入的`token`是全新的，没有和其他`token`计算过的！因此，如果重新计算所有`in_tokens`是对计算的极大浪费，需要把之前的计算缓存起来，即缓存下来Transformer所有层（MHA部分）计算后的`key`和`value`，即KV Cache！

:::tip
为什么不缓存`query`呢？
:::


## Transformer Inference Arithmetic
[分析transformer模型的参数量、计算量、中间激活、KV cache](https://zhuanlan.zhihu.com/p/624740065)
[Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)