---
title: "GPUDrive"
description: "gpudrive driving policy by self-play"
publishDate: "14 April 2025"
tags: ["tech/rl"]
---

> 苹果3月份的一篇[工作](https://arxiv.org/abs/2502.03349)(后面称为gigaflow工作)通过大规模强化学习Self Play，在不依赖任何人类驾驶数据的情况下，训练得到的驾驶策略函数在各大数据集中零样本测试表现SOTA。然而，论文中的关键GIGAFLOW模拟器并没有开源，好在有开源平替[gpudrive](https://github.com/Emerge-Lab/gpudrive)，可以用来玩一下。

## GPUDrive
GPUDrive在论文[GPUDrive: Data-driven, multi-agent driving simulation at 1 million FPS](https://arxiv.org/abs/2408.01584)，并开源。

### GPUDrive Code
分析一下GPUDrive的代码。

## Reward Design
奖励函数对驾驶行为的塑造非常重要，gigaflow的奖励函数设计。
