---
title: "RL 101"
description: "reinforcement learning basics"
publishDate: "3 April 2025"
tags: ["tech/rl"]
draft: false
---

> 重新拾起Reinforcement Learning的基础概念和算法。

## Terminology
强化学习中有很多的术语和概念，初学经常被搞得很懵逼，所以先从术语开始。

> 强化学习中有很多概念是一个概率密度函数，充满随机性。

### Basics
**Agent:**

**State $s_t$** 

**Action $a_t$**

**Policy $\pi$**

Policy function $\pi$

**Reward**

**State Transition**

**Return**

Return的定义是累积的未来回报(Cumulative **future** reward)，注意是**未来**:
$$
U_t=R_t+R_{t+1}+R_{t+2}+R_{t+3}+\cdots
$$

由于未来时刻$t$的奖励和当前比不一定等价，所以打个折扣，也就是Discounted return using $\gamma$:
$$
U_t=R_t+\gamma R_{t+1}+\gamma^2R_{t+2}+\gamma^3R_{t+3}+\cdots
$$

需要注意的是$U_t$是一个随机变量，随机性的来源有：


### Value Functions

#### Action-Value Function
Action-Value Function $Q_{\pi}(s, a)$ for policy $\pi$:

$$
Q_{\pi}(s, a)=\mathbb{E}(U_t \vert S_t=s_t, A_t=a_t)
$$

#### State-Value Function
$V$

## Valued-based RL

### DQN and TD
DQN是经典的Value-based的

### Temporal Difference (TD) Learning
训练DQN的方法

## Policy-based RL

### TRPO

### PPO
了解PPO需要几个背景知识：Importance Sampling, 

:::tip
**重要性采样(Importance Sampling)**

Importance Sampling是一种估计目标分布期望的技巧。当无法直接从目标分布$p(x)$时，通过另一个提议分布$q(x)$生成样本，并使用权重$\frac{p(x)}{q(x)}$修正期望值：
$$
\mathbb{E}_{x \sim p}[f(x)] = \mathbb{x \sim q} \big[f(x) \frac{p(x)}{q(x)} \big]
$$
:::

## Actor-Critic RL
