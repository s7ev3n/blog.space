---
title: "Variational Autoencoder"
description: "vae"
publishDate: "5 Feb 2025"
tags: ["tech/generative"]
---

## Background

### Notation
- 大写字母$X$表示随机变量
- 大写$P(X)$表示随机变量的概率分布
- $x\sim P(X)$表示从概率分布中采样出某个值$x$
- 小写$p(X)$表示随机变量$X$的概率密度函数
- $p(X=x)$表示某一点$x$处的概率值

### Bayesian
贝叶斯公式：
$$
p(Z|X)=\frac{p(X|Z)p(Z)}{p(X)}
$$
其中，$p(Z|X)$被称为后验概率(posterior probability)，$p(X|Z)$被称为似然函数(likelihood)，$p(Z)$被称为先验概率(prior probability)，$p(X)$是边缘似然概率。

### Likelihood
> 定义：The likelihood function (often simply called the likelihood) describes the joint probability of the observed data as a function of the parameters of the chosen statistical model. For each specific parameter value $\theta$  in the parameter space, the likelihood function $p(X | \theta)$ therefore assigns a probabilistic prediction to the observed data $X$.

总结一下上面定义的关键点：
- **似然不是概率**，尽管它看起来像个概率分布，它的加和不是$1$，似然$p(X|Z)$是关于$Z$的函数，是给定$X$变化$Z$的概率
- 一般情况，我们选定似然函数的形式，比如假设$Z$就是高斯分布的$\mu,\sigma$，我们来变换不同的$\mu, \sigma$，得到不同参数分布下得到$X$的概率，换句话说，一般似然函数是可计算的
- **和条件概率的区别**:哪个变量是固定哪个变量是变化的：条件概率$p(X|Z)$中$Z$是固定的，$X$是变化的；似然函数$p(X|Z)$，正相反，$X$是固定的，$Z$是变化的。

### Expecation
离散的随机变量$z$，服从分布$z \sim Q_{\phi}(Z|X)$(注意这个概率分布假定$X$是给定的，$Z$是变化的)。关于随机变量$x$的函数$f(z)$的期望定义：
$$
\mathbb{E}_{z \sim Q_{\phi}(Z|X)}[f(z)]=\sum_{z} q_{\phi}(z|x)\cdot f(z)
$$
其中，$q_{\phi}(z|x)$是取到某个$z$的概率值。另外，$\sum_{z} q_{\phi}(z|x)=1$

## Variational Inference
**变分推断(Variational Inference)是指通过优化方法“近似”后验分布的过程**。

我们都有后验概率的解析式$p(z|x)=\frac{p(x|z)p(z)}{p(x)}$，为什么不能直接计算后验概率而要近似呢？

依次来看似然$p(x|z)$，先验$p(z)$和边缘似然$p(x)$：
- 似然$p(x|z)$: 一般是假设似然函数的形式的，例如高斯分布
- 先验$p(z)$: 一般也可以估计，例如统计大量数据中猫的图片的数量占比，即为先验
- 边缘似然$p(x)$: 在高维的情况下，计算边缘似然$p(x)$是非常困难的

边缘似然$p(x)$的定义是$p(x)=\sum_{z}p(x|z)p(z)$，如果$z$是高维度向量$z=(z_1,z_2,...,z_n)$，每个维度$z_i$有非常多可能的取值，要遍历所有的$z$，计算$p(x)$是非指数级时间复杂度。

因此，变分推断通过引入一个参数化的近似分布$q_{\phi}(z)$，通过优化方法是其逼近真实的后验分布$p(z|x)$。
### Evidence Lower Bound (ELBO)
以下的推导来自这篇博客[^1]。我们定义一个参数化的分布$Q_{\phi}(Z|X)$(例如是高斯分布)来近似$P(Z|X)$

[^1]: [A Beginner's Guide to Variational Methods: Mean-Field Approximation](https://blog.evjang.com/2016/08/variational-bayes.html)

### Forward KL vs. Reverse KL


## Variational Autoencoder
Lil'Log的文章[^2]比较了多种Autoencoder，同时也包含VAE。
[^2]: [From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/)
