---
title: Loss functions
description: different loss functions
publishDate: "1 Jan 2023"
tags: ["tech/ml"]
---

> 这里总结下各种遇到的有用的Loss Functions.

## Classification Loss Functions

### Cross Entropy and KL Divergence

### Binary Cross-Entropy Loss
BCE又称为Sigmoid Cross-Entropy Loss，适用于二分类任务，每个样本只有两个类别：正例和负例，公式：
$$
BCE=-\frac{1}{N}\sum_{i=1}^{N}[y_i log(p_i) + (1-y_i)log(1-p_i)]
$$
其中，$y_i$是真实标签，即$0$或者$1$，$p_i$是模型预测为$1$的概率（$p_i$是使用Sigmoid函数计算），$N$是样本数量。

代码上，


### Categorical Cross-Entropy loss

### Focal Loss
Focal Loss是计算机视觉中用于处理分类问题中类别不平的情况，即如果一个样本被模型高概率预测为正确，那么它对loss的贡献应该很小，而一个样本如果被模型预测错误，那么它对loss的贡献应该更大，即使模型更关注难样本。

Focal Loss使用Sigmoid函数，也应该被认为是BCE Loss。

$$
FL(p_t)=-\alpha_t(1-p_t)^{\gamma}log(p_t)
$$

代码实现：
```python
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float() # (N, C) N表示可以把BSZ等等维度合并的最终维度
    targets = targets.float() # (N,) 表示N个样本的类别，是0还是1
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
```


[^1]: [Understanding Categorical Cross-Entropy Loss, Binary Cross-Entropy Loss, Softmax Loss, Logistic Loss, Focal Loss](https://gombru.github.io/2018/05/23/cross_entropy_loss/)

## Ranking Loss Functions

[^2]: [Understanding Ranking Loss, Contrastive Loss, Margin Loss, Triplet Loss, Hinge Loss](https://gombru.github.io/2019/04/03/ranking_loss/)