---
title: "End to End Autonomous Driving"
description: "E2E AD"
publishDate: "1 March 2025"
tags: ["tech/adas/e2e"]
---

> 早在2016年读博期间，尝试过古早的行为克隆端到端(End to End)驾驶[End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316)和强化学习端到端[DDPG Torcs Racing](https://github.com/s7ev3n/gym_torcs_modified)。再Tesla FSD的推动下，智驾行业技术范式正在向极致数据驱动的端到端发展。这篇博客尝试对目前为止端到端范式的技术思想、最新进展做持续的思考和更新。

## High Level
当前(2025.03)绝大部分的E2E AD都采用了模仿学习(Imitation Learning, IL)，即使用大量的人类驾驶数据为监督学习驾驶行为，使用IL学习需要闭环反馈的驾驶行为存在很多问题。比较少数使用Model Based IL来同时学习World Model和Driving Policy[^1]。

:::tip
定义和区分一下Imitation Learning(IL)和Behavior Cloning(BC)这两个常见的概念。

**Imitation Learning(IL)**: 是一种通过模仿专家行为来训练智能体的方法，属于强化学习的子领域。其核心是让智能体从专家示范（如人类行为）中学习策略，而非依赖传统的奖励函数。

**Behavior Cloning(BC)**: 是IL的一种具体实现方式，属于监督学习框架。它通过直接复制专家在特定状态下的动作来学习策略，即输入为状态（如传感器数据），输出为动作（如方向盘转角），训练目标是最小化预测动作与专家动作的差异。

| **维度**               | **Behavior Cloning**  | **Imitation Learning**   |
|-----------------------|-----------------------|------------------------------------------------|
| **方法本质**           | 监督学习，直接映射状态到动作。          | 可能结合强化学习、在线交互或奖励函数推断。     |
| **数据依赖**           | 依赖静态专家数据集，无环境交互。        | 可能动态与环境交互（如DAgger）或推断奖励函数。 |
| **泛化能力**           | 在训练数据分布内表现好，分布外易失效。  | 通过推断专家意图或在线修正，泛化能力更强。     |
| **累积误差**           | 易因状态偏移（distribution shift）导致错误累积。 | 通过主动探索（如DAgger）或优化策略减少误差。   |
| **典型算法**           | 简单的监督回归/分类。                   | 逆强化学习（IRL）、DAgger、对抗模仿学习（GAIL）。 |
:::

Wang Naiyan在知乎回答[^2]中对IL的问题做了一些分析，主要存在两个问题：Out of Distribution (OOD)和Sparse Supervision。我搬运放在这里。


[^1]: [Model-based imitation learning for urban driving](https://arxiv.org/abs/2210.07729)
[^2]: [Imitation Learning or not?](https://zhuanlan.zhihu.com/p/721582016?utm_id=0)

Out of Distribution (OOD)：这个问题很早就熟悉，即驾驶策略的泛化性问题：“由于时序的累计，误差由于修正不及时可能会导致最后巨大的偏差，也就是会让系统进入到一个训练数据中不常见的state，也就是所谓的compounding error问题”。

“为了缓解这个问题，一个直观的做法是对数据做augmentation，这也是目前端到端方法训练中常用的trick。但是这不能从根本上解决这样的问题，理论完善的方法，类似于DAgger这种需要在线access expert的方法在自动驾驶场景中又无法低成本实现。更为重要的是，由于p(a | s)的高度多峰性，为了能学习到，数据采集中不仅仅要覆盖到罕见的state，还要覆盖到这样state下的每一个峰值，这使得本来就很难的采集变得雪上加霜。”

“一个常见的办法是结合RL和IL，也就是说会设计一些reward或者cost，让系统在这样的情况下回归到in distribution的状态”


Sparse Supervision: 可以区分IL和RL的奖励稀疏问题。IL中："监督稀疏性主要来源于问题本身是一个非常高维到低维的映射，然后监督信息的信息量明显不足，即只提供一个正样本作为监督。在端到端自动驾驶中，输入往往是一个多帧多视角的视频序列，输出只有一条轨迹（往往使用一个十几维的参数化形式）。这使得数据利用效率非常之低。"

在RL中：“稀疏监督的问题在于，虽然我们可以设计种类繁多的reward，但是reward往往只会在terminal state上给出，我们需要漫长的propagation的过程，才能使这样监督信息传递到其他的state中去。这个问题被讨论的非常多，就不再赘述。比如，在自动驾驶中，我们往往会根据horizon内轨迹的碰撞设计cost，但是我们希望的驾驶行为是在进入到这样的危险状况之前，就采取防御性驾驶行为规避进入到这样状态的可能性。”

如何把稀疏的奖励信号变得Dense？“一个常见的思路是针对任务手工设计dense reward”。但是这个思路其实很难。

“另外一个思路是通过大模型将世界常识注入到IL中，来解释为什么expert会采用这样的一个action。也就是通过大模型将demonstration拆解并提供更多的监督。”

