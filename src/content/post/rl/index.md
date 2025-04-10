---
title: "RL 101"
description: "reinforcement learning basics"
publishDate: "3 April 2024"
tags: ["tech/rl"]
draft: false
---

> 重新拾起Reinforcement Learning的基础概念和算法。

## Terminology
强化学习中有很多的术语，所以先从术语开始。

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


## Value Functions

### Action-Value Function
Action-Value Function $Q_{\pi}(s, a)$ for policy $\pi$:

$$
Q_{\pi}(s, a)=\mathbb{E}(U_t \vert S_t=s_t, A_t=a_t)
$$

### State-Value Function


## Policy-based

