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
MCTS就是建立一颗树的过程，我的理解这像是一个强化学习训练过程，从一个给定的状态出发开始“可回溯”的探索，环境会在达到某个状态时给出奖励。
举下棋的例子，你处在某一个棋局状态之中，应该怎么走呢？如果之前见过一样的棋局状态，就知道下面走哪一步胜率最高，如果不知道，那就在头脑中模拟，头脑中模拟就是MCTS过程，**没有真实的下棋子，没有改变任何的棋局状态**，通过MCTS后，我知道最好的下一步，这时候真正下棋子，会到达新的棋局状态，然后在新的棋局状态开始新一轮MCTS。它的逻辑可以用伪代码表示：

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

> 累积评分值$v_i$是MCTS迭代之后，用于最终决定执行动作的依据，选择$v_i$最大的动作执行。

#### 选择 Selection
在选择阶段，根据当前的节点状态，决定如何向下探索，对应流程图中的第一个选择块（判断是否是叶子节点）：

1. 该节点的所有可行动作都已经被探索过，依据UCB值(Upper Confidence Bounds)选择最大值的节点
2. 该节点有可行动作还未被探索