---
title: Monte Carlo Tree Search
description: mcts
publishDate: "4 May 2025"
updatedDate: "4 May 2025"
tags: ["tech/ml"]
---

## Monte Carlo Tree Search

蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）是一种用于在某些类型的决策过程中（尤其是在具有巨大搜索空间的问题中，如棋类游戏）寻找最优决策的启发式搜索算法，对比之下DFS和BFS就是暴力搜索。MCTS被用在AlphaGo, MuZero中，近期的LLM-Reasoning模型也有使用。

### 概述
MCTS就是建立一颗树的过程，从一个给定的状态出发开始“可回溯”的探索，环境会在达到某个状态时给出奖励，然后回溯经过的节点更新节点的一些特征值，我肤浅理解这有些像是一个强化学习训练过程。

以下棋为例，你处在某一个棋局状态之中，应该怎么走呢？如果之前见过一样的或类似的棋局状态，就知道下面走哪一步胜率最高，如果不知道，那就在头脑中模拟，头脑中模拟就是MCTS过程，**没有真实的下棋子，没有改变任何的棋局状态**，通过MCTS后，就知道最好的下一步，这时候才真正下棋子，到达新的棋局状态，然后在新的棋局状态开始新一轮MCTS，只要见过海量的棋局状态，经过海量的MCTS，就可以知道最好的下一步。它的逻辑可以用伪代码表示：

```python
tree = MCTS()
board = new_board()
while True:
    # 输入你的下棋的棋子
    row_col = input("enter row,col: ")
    row, col = map(int, row_col.split(","))
    index = 3 * (row - 1) + (col - 1)
    if board.tup[index] is not None:
        raise RuntimeError("Invalid move")
    # 下棋（棋盘改变状态）
    board = board.make_move(index)
    print(board.to_pretty_string())
    # 如果棋盘到终止状态，游戏结束
    if board.terminal:
        break
    # 使用mcts的四步骤进行迭代
    for _ in range(mcts_iterations):
        tree.do_mcts(board)
    # mcts过后，我们就知道了最优的下一步，选择执行下一步，更新棋盘
    board = tree.choose(board)
```

### MCTS算法流程
MCTS的迭代核心有四个步骤：选择(Selection) $\rightarrow$ 扩展(Expansion) $\rightarrow$ 模拟(Simulation/Rollout) $\rightarrow$ 回溯(Backpropagation/Update)，如下面的流程图所示：
<div class="mermaid">
graph LR
    B["根节点S<sub>0</sub>"] --> C{"当前节点是<br/>叶子节点?"};
    C -- 否 --> D["**选择(Selection)** <br/>当前节点 = 当前节点中UCB(S<sub>i</sub>)最大的子节点"];
    D --> C;
    C -- 是 --> E{"当前节点的<br/>n<sub>i</sub>=0?"};
    E -- 是 --> F["**模拟(Rollout)**"];
    E -- 否 --> G["**扩展(Expansion)** <br/>为当前节点的每个可用动作向树中添加新状态"];
    G --> H["当前 = 第一个<br/>新子节点"];
    H --> I["**模拟(Rollout)**"];
    J["**反向传播(Backpropagation)** <br/>用模拟结果更新自当前叶节点至根S<sub>0</sub>路径上各节点的N、Q值"];
    F --> J;
    I --> J;
    J -.-> B;
</div>

#### 节点
每个节点需要记录三个基本信息：
- 当前节点的状态，例如棋盘的棋局
- 该节点被访问的次数$n_i$
- 累积评分值$v_i$，是平均奖励值，即获得的总奖励值除以$n_i$

> 累积评分值$v_i$是MCTS迭代之后，用于最终决定执行动作的依据，选择$v_i$最大的动作执行。是MCTS外部的游戏棋局选择执行下一步的依据。作为对比，UCB值不需要存储，只发生在MCTS的选择阶段，用于推进树的向下搜索。

#### 选择 Selection
选择(Selection)阶段的目标是找到下一个节点，以便进行扩展(Expansion)。被选中进行扩展的节点必须是“未完全扩展的”，即它仍有未尝试的动作可以执行。在选择阶段，可能遇到下面的情况：

1. 该节点的所有可行动作都已经被探索过，即该节点被完全探索过，依据UCB值(Upper Confidence Bounds)选择最大值的节点往下搜索，因为没有找到“未完全扩展”的节点，所以选择并没有结束
2. 该节点有可行动作还未被探索过，此时选择阶段结束，这个节点就是要送到扩展阶段的节点
3. 该节点是终止节点

#### 扩展
扩展阶段的输入是选择阶段“选中”的未完全扩展的节点，扩展阶段的目标是从当前节点的状态下的可用的、未使用的动作列表中选择一个动作，并作为一个新节点添加到当前节点下作为子节点。

#### 模拟
模拟阶段从当前给定的节点出发，使用默认策略从合法动作中选择一个动作，推进更新到下一个状态，直到快速的玩完游戏获得奖励，例如获胜、输掉或者平局等。**需要注意的是模拟阶段并添加任何新的节点，它只是想要从当前阶段快速的玩完游戏获得结果**，从而进入反向传播。

#### 反向传播
反向传播阶段就是游戏到了终点，得到了奖励，把奖励值加到经过的每一个节点上，即更新每个节点的总奖励值。因为模拟阶段并没有创建新的节点，所以奖励值将首先加到模拟开始的节点上，然后沿着选择的路径直到根节点。
