---
title: "E2E Autonomous Driving Lit Review"
description: "E2E AD Lit Review"
publishDate: "1 March 2025"
tags: ["tech/research", "tech/adas"]
---

> 持续记录端到端自动驾驶的论文和认识。

## Insights
- [2025.06]Waymo的论文Scaling Laws of Motion Forecasting and Planning[^3]中使用了非常惊人的5000000小时的数据进行了IL的训练，远远超过开源数据集的训练量，模型的能力持续提升。IL中的OOD问题，是不是因为训练数据的投入远远不够？像LLM范式中的**Heavy** IL + RL finetune是否能在AI Planning中同样成功？
- [2025.03]当前绝大部分的E2E AD都采用了模仿学习(Imitation Learning, IL)，即使用大量的人类驾驶数据为监督学习驾驶行为，使用IL学习需要闭环反馈的驾驶行为存在很多问题。比较少数使用Model Based IL来同时学习World Model和Driving Policy[^1]。采用IL的模型普遍会遇到比较严重的泛化问题。
- Wang Naiyan在知乎回答 [^2] 中对IL E2E问题的分析 
    <details>
    <summary>主要存在两个问题：Out of Distribution (OOD)和Sparse Supervision：</summary>

    1. Out of Distribution (OOD)：这个问题很早就熟悉，即驾驶策略的泛化性问题：“由于时序的累计，误差由于修正不及时可能会导致最后巨大的偏差，也就是会让系统进入到一个训练数据中不常见的state，也就是所谓的compounding error问题”。
    
        为了缓解这个问题，一个直观的做法是对数据做augmentation，这也是目前端到端方法训练中常用的trick。但是这不能从根本上解决这样的问题，理论完善的方法，类似于DAgger这种需要在线access expert的方法在自动驾驶场景中又无法低成本实现。更为重要的是，由于$p(a \vert s)$的高度多峰性，为了能学习到，数据采集中不仅仅要覆盖到罕见的state，还要覆盖到这样state下的每一个峰值，这使得本来就很难的采集变得雪上加霜。
        
        一个常见的办法是**结合RL和IL**，也就是说会设计一些reward或者cost，让系统在这样的情况下回归到in distribution的状态。

    2. Sparse Supervision: 可以区分IL和RL的奖励稀疏问题。IL中："监督稀疏性主要来源于问题本身是一个非常高维到低维的映射，然后监督信息的信息量明显不足，即只提供一个正样本作为监督。在端到端自动驾驶中，输入往往是一个多帧多视角的视频序列，输出只有一条轨迹（往往使用一个十几维的参数化形式）。这使得数据利用效率非常之低。"
        
        在RL中：“稀疏监督的问题在于，虽然我们可以设计种类繁多的reward，但是reward往往只会在terminal state上给出，我们需要漫长的propagation的过程，才能使这样监督信息传递到其他的state中去。这个问题被讨论的非常多，就不再赘述。比如，在自动驾驶中，我们往往会根据horizon内轨迹的碰撞设计cost，但是我们希望的驾驶行为是在进入到这样的危险状况之前，就采取防御性驾驶行为规避进入到这样状态的可能性。”

        如何把稀疏的奖励信号变得Dense？
       - “一个常见的思路是针对任务手工设计dense reward”。但是这个思路其实很难。
       - 另外一个思路是通过大模型将世界常识注入到IL中，来解释为什么expert会采用这样的一个action。也就是通过大模型将demonstration拆解并提供更多的监督。”
    </details>

[^1]: [Model-based imitation learning for urban driving](https://arxiv.org/abs/2210.07729)
[^2]: [Imitation Learning or not?](https://zhuanlan.zhihu.com/p/721582016?utm_id=0)
[^3]: [Scaling Laws of Motion Forecasting and Planning](https://arxiv.org/pdf/2506.08228)

## Lit Review

- [Trajeglish: Traffic Modeling as Next-Token Prediction](https://arxiv.org/abs/2312.04535)
  - TL;DR: Trajeglish对所有Agent运动状态的相对转移进行聚类得到了Motion Vocabulary，这样轨迹就可以离散为Token的序列。Motion Vocabulary成为后续工作SMART的离散化方法。
- [MotionLM: Multi-Agent Motion Forecasting as Language Modeling](https://arxiv.org/abs/2309.16534)
  - TL;DR: MotionLM同样是使用NTP来构建所有Agent的运动，它采用的是x和y轴均匀的离散化方法。
