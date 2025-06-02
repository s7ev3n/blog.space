---
title: Monte Carlo Tree Search
description: mcts
publishDate: "4 May 2025"
updatedDate: "4 May 2025"
tags: ["tech/ml"]
---

## Monte Carlo Tree Search

蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）是一种用于在某些类型的决策过程中（尤其是在具有巨大搜索空间的问题中，如棋类游戏）寻找最优决策的启发式搜索算法。MCTS被用在AlphaGo, MuZero中，近期的LLM-Reasoning模型也有使用。

### 大白话
MCTS是一个迭代的过程，或者说训练的过程，比如你在下棋处在某一个棋局状态之中，我应该怎么走呢？如果我之前见过一样的棋局状态，我就知道下面走哪一步胜率最高，如果我不知道，那我就在头脑中模拟，头脑中模拟就是MCTS的迭代过程，**没有真实的下棋子**，通过MCTS迭代，我知道最好的下一步，这时候下棋子。下棋子之后，会到达新的棋局状态，然后循环迭代。MCTS是一种在交互中在线学习的方法，并不是监督学习的方法。

它的伪代码大约是：
```python
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
    # 如果棋盘到终止状态，结束
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
    C -- 否 --> D["**选择(Selection)**<br/>当前节点 = 当前节点中UCB1(S<sub>i</sub>)最大的子节点"];
    D --> C;
    C -- 是 --> E{"当前节点的<br/>n<sub>i</sub>=0?"};
    E -- 是 --> F["**模拟(Rollout)**"];
    E -- 否 --> G["**扩展(Expansion)**<br/>为当前节点的每个可用动作向树中添加新状态"];
    G --> H["当前 = 第一个<br/>新子节点"];
    H --> I["**模拟(Rollout)**"];
    J["**反向传播(Backpropagation)**<br/>用模拟结果更新自当前叶节点至根S<sub>0</sub>路径上各节点的N、Q值"];
    F --> J;
    I --> J;
    J -.-> B;
</div>
