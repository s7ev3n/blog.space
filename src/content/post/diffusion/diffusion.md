---
title: "Variational Autoencoder"
description: "vae"
publishDate: "3 Jan 2025"
tags: ["tech/diffusion"]
---

## Bayesian
贝叶斯公式：
$$
p(Z|X)=\frac{p(X|Z)p(Z)}{p(X)}
$$
其中，$p(Z|X)$被称为后验概率(posterior probability)，$p(X|Z)$被称为似然函数(likelihood)，$p(Z)$被称为先验概率(prior probability),$p(X)$是边缘概率。

假设$X$表示一张图像，$Z$表示类别是否是一只猫，$p(Z|X)$表示给定图片$X$是猫的概率。

### Likelihood
**似然不是概率**，尽管它看起来像个概率分布，它的加和不是$1$，似然是关于$Z$的函数。

**和条件概率的区别**:哪个变量是固定哪个变量是变化的：条件概率$p(X|Z)$中$Z$是固定的，$X$是变化的；似然函数$p(X|Z)$，正相反，$X$是固定的，$Z$是变化的。

### Expecation
对于离散的随机变量$z$，服从分布$z \sim Q_{\phi}(Z|X)$(注意这个概率分布假定$X$是给定的，$Z$是变化的)。对函数$f(z)$的期望定义为
$$
\mathbb{E}_{z \sim Q_{\phi}(Z|X)}[f(z)]=\sum_{z} q_{\phi}(z|x)\cdot f(z)
$$
其中，$q_{\phi}(z|x)$是取到某个$z$的概率值。另外，$\sum_{z} q_{\phi}(z|x)=1$

## Variational Methods
这篇博客[^1]很清晰的介绍了变分方法，和VAE中基本最重要的变分下界(ELBO)。


[^1]: [A Beginner's Guide to Variational Methods: Mean-Field Approximation](https://blog.evjang.com/2016/08/variational-bayes.html)

## Variational Autoencoder
Lil'Log的文章[^2]比较了多种Autoencoder，同时也包含VAE。
[^2]: [From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/)
