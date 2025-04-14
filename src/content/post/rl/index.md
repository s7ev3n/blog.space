---
title: "RL 101"
description: "reinforcement learning basics"
publishDate: "3 April 2025"
tags: ["tech/rl"]
draft: false
---

> 重新拾起Reinforcement Learning的基础概念和算法。

## Terminology
强化学习中有很多的术语和概念，初学经常被搞得很懵逼，所以先从术语开始，把基础的概念理解好。

公式中符号：$X$表示随机变量，$x$表示随机变量$X$的观测值，$P$概率密度函数，$\mathcal{S}$表示集合。

> 强化学习框架中充满了随机性，有很多概念是一个概率密度函数，因此采样方法，例如MC采样和求期望是常见的操作。

### Basics
**Agent**
Agent是环境中活动（执行动作）的主体，例如在真实世界中行驶的自动驾驶车（Ego Vehicle）。

**Environment**
环境（Envionment）或者现在的有些说法叫世界模型（World Model），是Agent所活动的环境的抽象。

**State $S$** 
状态（state）指的是当前Agent所包含的信息，这些信息决定着Agent的未来。在某时刻$t$的状态$S_t$是一个随机变量，$s_t$是当前时刻的观测值，可以有很多可能的$s_t$。所有可能的状态的集合称为状态空间$\mathcal{S}$，即State Space。

**Action $A$**
动作（action）指的是Agent采取的动作。在某时刻$t$的状态$A_t$是一个随机变量，$a_t$是当前时刻的观测值，可以有很多可能的$a_t$，这些动作可能是离散值，也可能是连续值。所有可能的动作的集合称为动作空间$\mathcal{A}$，即Action Space。

**Reward $R$**
奖励（reward）指的是Agent采取了动作环境给予Agent的奖励值。在某时刻$t$的状态$R_t$是一个随机变量，$r_t$是当前时刻的观测值。

奖励也会被称为奖励函数（reward function），环境根据当前的状态$s_t$和采取的不同动作$a_t$，会有不同的奖励值。

奖励也是一个很困难的话题，因为强化学习的目标是**最大化**未来的奖励，它塑造了Agent的学习。
它可以是根据某个任务定义的，例如AlphaGo下棋，赢了得到了价值100的奖励，输了要惩罚100，这里奖励值的确定并没有科学的依据。
例如在大语言模型应用强化学习进行RLHF，最大化的奖励是和人类对齐(alignment)的回答，但是模型也会出现Reward Hacking[^1]。

RL算法也经常会面临奖励稀疏(Sparse Reward)的问题，导致RL比较大的问题是学习的效率不高(inefficient)的问题，即需要超级大量的试错才能学到有效的动作。

[^1]: [Reward Hacking in Reinforcement Learning](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)

**Return $U$**
回报Return的定义是累积的**未来**回报(Cumulative **future** reward)，注意是**未来**:
$$
U_t=R_t+R_{t+1}+R_{t+2}+R_{t+3}+\cdots
$$

由于未来时刻$t$的奖励和当前比不一定等价，所以打个折扣，也就是Discounted return using $\gamma$:
$$
U_t=R_t+\gamma R_{t+1}+\gamma^2R_{t+2}+\gamma^3R_{t+3}+\cdots
$$

需要注意的是$U_t$是一个随机变量，它依赖于未来未观测到的奖励，而这些奖励依赖于未来采取的动作和状态，但是回报可以通过积分掉未观测到的变量求**期望**。

**Trajectory** 
轨迹是Agent与环境交互的序列：$s_1, a_1, r_1, s_2, a_2, r_2, \cdots$

**State Transition**
状态转移（state transition）$p(\cdot \vert s, a)$指的是根据当前Agent的状态$s$和采取的动作$a$，环境转移到新的状态$s'$的概率，因此状态转移时一个概率密度函数$p(s'\vert s,a)=P(S'=s' \vert S=s, A=a)$。

### Policy Function
策略函数(Policy function)$\pi$是Agent当前的状态$s$映射到动作空间$\mathcal{A}$内所有动作的概率分布，因此它是一个概率密度函数:$\pi(a \vert s)=P(A=a \vert S=s)$，所以有$\sum_{a \in \mathcal{A}}\pi(a \vert s)=1$。

### Value Functions

#### Action-Value Function
动作价值函数(Action-Value Function) $Q_{\pi}(s, a)$，即经常见到的$Q$函数，描述了在给定状态$s$下采取某个动作$a$的好坏，这个好坏是通过回报$U_t$的期望来进行评价的，:
$$
Q_{\pi}(s, a)=\mathbb{E}(U_t \vert S_t=s_t, A_t=a_t)
$$

为什么是期望呢？因为未来（$t+1$时刻之后）可能采取的动作和进入的状态都是随机变量，但是我们通过求期望，即通过概率加权求和或积分消掉未来的随机变量，从而大体知道可以预期的平均回报是多少。

$Q$函数依赖策略函数$\pi(a \vert s)$和状态转移函数$p(\cdot \vert s, a)$。

**最佳动作价值函数(Optimal action-value function)** $Q^*(s, a)$衡量在不同的策略函数$\pi$和状态$s_t$下，采取动作$a_t$的好坏： 
$$
Q^*(s, a)=\underset{\pi}{\text{max}}Q_{\pi}(s,a)
$$
注意，$Q^*$和策略函数$\pi$没有关系。

#### State-Value Function
状态价值函数$V_{\pi}$衡量给定策略$\pi$，当前状态的好坏，相当于对动作价值函数积分掉所有的动作：
$$
V_{\pi}=\mathbb{E}_{A\sim \pi(\cdot \vert s_t)}\big[Q_{\pi}(s_t, A) \big]=\int_{\mathcal{A}} \pi(a\vert s_t) \cdot Q_{\pi}(s_t,a) \, da
$$
状态价值函数描述了给定策略$\pi$现在所处的状态的好坏，不管采取什么状态。

**最佳状态价值函数(Optimal state-value function)** $V^*(s_t)$衡量在不同的策略函数$\pi$中当前状态$s_t$的好坏：
$$
V^{*}(s) = \underset{\pi}{\text{max}}V_{\pi}(s)
$$
注意，$V^*$和策略函数$\pi$没有关系。

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
