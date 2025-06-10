---
title: Monte Carlo Tree Search
description: mcts
publishDate: "4 May 2025"
updatedDate: "4 May 2025"
tags: ["tech/gems"]
---

## Monte Carlo Tree Search

蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）是一种用于在某些类型的决策过程中（尤其是在具有巨大搜索空间的问题中，如棋类游戏）寻找最优决策的启发式搜索算法，对比之下DFS和BFS就是暴力搜索。MCTS被用在AlphaGo, MuZero中，近期的LLM-Reasoning模型也有使用。

### 概述
使用MCTS解决某一个问题（例如下棋）就是建立求解这个问题的一颗搜索树的过程，从一个给定的状态出发开始，环境会在达到某个终止状态时给出奖励，然后回溯路径上的节点，并更新节点的一些特征值。

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
        tree.run_mcts(board)
    # mcts过后，我们就知道了最优的下一步，选择执行下一步，更新棋盘
    board = tree.choose(board)
```

### MCTS算法流程
MCTS的迭代核心有四个步骤：选择(Selection) $\rightarrow$ 扩展(Expansion) $\rightarrow$ 模拟(Simulation/Rollout) $\rightarrow$ 回溯(Backpropagation/Update)。
:::tip
在看其他资料的时候，有些资料的表示并不一致，造成了理解的困难：
- 未完全扩展节点，指的是该节点存在子节点，但是该节点还有未探索的可行子节点没有在作为节点添加。实际上，这和扩展的方式有关，常见的实现会一次性添加所有的可行节点，就不会出现这种情况了
- 叶子节点。叶子节点的定义按照尝试理解为没有子节点的节点，但是在MCTS的语境下，“未完全扩展节点”也属于叶子节点，即不是完全扩展节点或终止节点都算是叶子节点。这种说法显然有很大的歧义
- UCT的计算在不同的扩展方式下也不同。如果扩展阶段一次添加一个节点，那么UCT计算发生在完全扩展节点，因为在未完全展开节点，会结束选择阶段，进入扩展阶段。如果一次扩展所有节点，UCT计算每到一个节点都会发生，未探索的节点（一定是叶子节点）UCT值无穷大，因此每次都会选择未探索的节点

扩展方式的不同会造成选择阶段一些行为的不同，本文全部按照“**扩展阶段一次性添加所有的可行节点**”来理解。
:::
如下面的流程图所示：
<div class="mermaid">
graph LR
    B["根节点S<sub>0</sub>"] --> C{"当前节点是<br/>叶子节点?"};
    C -- 否 --> D["**选择(Selection)** <br/>当前节点 = 当前节点中UCB(S<sub>i</sub>)最大的子节点"];
    D --> C;
    C -- 是 --> E{"当前节点的<br/>n<sub>i</sub>=0?"};
    E -- 是 --> F["**模拟(Rollout)**"];
    E -- 否 --> G["**扩展(Expansion)** <br/>为当前节点一次性扩展所有可用动作作为子节点"];
    G --> H["随机选择一个子节点"];
    H --> I["**模拟(Rollout)**"];
    J["**反向传播(Backpropagation)** <br/>用模拟结果更新自当前叶节点至根S<sub>0</sub>路径上各节点的N、Q值"];
    F --> J;
    I --> J;
    J -.-> B;
</div>

也可以用如下的代码来理解：
```python
def run_mcts(self, node):
    "Make the tree one layer better. (Train for one iteration.)"
    path = self._select(node)
    leaf = path[-1]
    self._expand(leaf)
    reward = self._simulate(leaf)
    self._backpropagate(path, reward)

```

#### 节点
每个节点需要记录三个基本信息：
- 当前节点的状态，例如棋盘的棋局
- 该节点被访问的次数$n_i$
- 累积评分值$v_i$，是平均奖励值，即获得的总奖励值除以$n_i$

> 累积评分值$v_i$是MCTS迭代之后，用于最终决定执行动作的依据，选择$v_i$最大的动作执行。是MCTS外部的游戏棋局选择执行下一步的依据。作为对比，UCB值不需要存储，只发生在MCTS的选择阶段，用于推进树的向下搜索。

#### 选择 Selection
选择(Selection)阶段的目标是找到下一个节点来进行扩展(Expansion)。我们默认扩展阶段会一次性添加所有的可行动作作为子节点，所以选择阶段结束一定是发生在叶子节点。在选择阶段搜索过程中，可能遇到下面的情况：

1. 该节点是非叶子节点，依据UCB值(Upper Confidence Bounds)选择最大值的节点往下搜索，由于$n_i=0$时，UCB的值为无穷大，此时随机选择一个即可
2. 该节点是一个叶子节点，此时选择阶段结束。叶子节点就是没有子节点的节点，它可能有两种情况，这个节点被“探索”过，就是被模拟过，所以有过统计信息，例如被访问次数不为0，另一种情况是这个节点是扩展阶段添加的，但是还没有进行过模拟。

可以参考下面的代码理解：
```python
self.Q = defaultdict(int)  # total reward of each node
self.N = defaultdict(int)  # total visit count for each node
self.children = dict()  # 记录所有扩展过的节点，可以通过children.keys()来获得扩展过的子节点
# self.children[node]是当前节点的所有子节点
        
def _select(self, node):
    "Find an unexplored descendent of `node`"
    path = []
    while True:
        path.append(node)
        # node not in self.children 表示这个节点从未被扩展过
        # not self.children[node] 为True是表示当前节点是叶子节点
        if node not in self.children or not self.children[node]:
            # node is either unexplored or terminal
            return path
        # 从当前子节点中排出掉已经扩展过的节点就是未扩展的节点
        unexplored = self.children[node] - self.children.keys()
        if unexplored:
            # pop操作是随机选择一个未扩展子节点
            n = unexplored.pop()
            path.append(n)
            return path
        # 所有的子节点都被扩展过（都被self.children记录过），就UCT选择
        node = self._uct_select(node)  # descend a layer deeper
```

再来说一下UCT函数，它平衡了探索和利用：
$$ 
\text{UCT} = \frac{w_i}{n_i} + C \sqrt{\frac{\ln N}{n_i}} 
$$
其中，$n_i$是当前节点的访问次数，由于$n_i$是分母，所以一个新添加的节点的UCB的值是无穷大；$N$是当前节点父节点的访问次数，$C$是一个常数，$w_i$是当前节点获得的奖励。

```python
def _uct_select(self, node):
    "Select a child of node, balancing exploration & exploitation"

    # All children of node should already be expanded:
    assert all(n in self.children for n in self.children[node])

    log_N_vertex = math.log(self.N[node])

    def uct(n):
        "Upper confidence bound for trees"
        return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
            log_N_vertex / self.N[n]
        )

    return max(self.children[node], key=uct)
```

#### 扩展 Expansion
扩展阶段的输入是选择阶段“选中”的未完全扩展的节点，扩展阶段的目标是从当前节点的状态下的可用的、未使用的动作列表中选择一个动作，并作为一个新节点添加到当前节点下作为子节点。

> 有些说法会说，在扩展阶段添加一个未被探索的动作作为子节点，但是在实际实现中，这里选择一次性添加所有的可行动作:

```python
def _expand(self, node):
    "Update the `children` dict with the children of `node`"
    if node in self.children:
        return  # already expanded
    # add **all unexplored children** of `node` to the tree
    self.children[node] = node.find_children()
```

#### 模拟 Simulation Rollout
模拟阶段从当前给定的节点出发，使用默认策略从合法动作中选择一个动作，推进更新到下一个状态，直到快速的玩完游戏获得奖励，例如获胜、输掉或者平局等。**需要注意的是模拟阶段并不添加任何新的节点，它只是想要从当前阶段快速的玩完游戏获得结果**，所以只要random快速做出决策即可：

```python
def _simulate(self, node):
    "Returns the reward for a random simulation (to completion) of `node`"
    invert_reward = True
    while True:
        if node.is_terminal():
            reward = node.reward()
            return 1 - reward if invert_reward else reward
        node = node.find_random_child()
        invert_reward = not invert_reward

```

#### 反向传播 Backpropagate
反向传播阶段就是游戏到了终点，得到了奖励，把奖励值加到经过的每一个节点上，即更新每个节点的总奖励值。因为模拟阶段并没有创建新的节点，所以奖励值将首先加到模拟开始的节点上，然后沿着选择的路径直到根节点：

```python
def _backpropagate(self, path, reward):
    "Send the reward back up to the ancestors of the leaf"
    for node in reversed(path):
        self.N[node] += 1
        self.Q[node] += reward
        reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa
```

#### MCTS实现
代码的实现参考一份极简的开源实现[^1]
[^1]: [A minimal implementation of Monte Carlo tree search (MCTS)](https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1):

```python
from abc import ABC, abstractmethod
from collections import defaultdict
import math

class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        return max(self.children[node], key=score)

    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        # add all unexplored children of `node` to the tree
        self.children[node] = node.find_children()

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        invert_reward = True
        while True:
            if node.is_terminal():
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            node = node.find_random_child()
            invert_reward = not invert_reward

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)


class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True
```
