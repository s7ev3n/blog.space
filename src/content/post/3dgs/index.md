---
title: "3D Gaussian Splatting learning notes"
description: "3dgs learning notes and code implementation"
publishDate: "3 Dec 2024"
updatedDate: "18 Jan 2025"
coverImage:
  src: "./figs/3dgs_pipeline.svg"
  alt: "3DGS pipeline"
tags: ["research"]
draft: false
---

> 3D Gaussian Splatting(3DGS)由2023年论文[3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079)中提出，由于渲染速度快且质量高，迅速超越(NeRF)称为新视角渲染的热点，涌现出大量优秀的后续工作。本篇笔记以论文原文和原作者的[教程](https://3dgstutorial.github.io/3dv_part1.pdf)为基础，并包括一部份简单的渲染代码实现，没有训练代码。

## Background
对于不了解图形学的同学(包括我自己)，论文中有一些术语和技术值得简单介绍。

### What is Rasterization (删格化)?
栅格化（Rasterization）是将矢量图形转换为像素点阵的过程，是计算机图形学中的一个基础概念。简单来说，就是把连续的几何图形（如线段、多边形、曲线等）转换成离散的像素点来显示在屏幕上。Gaussian Splatting在最终显示时仍需要将结果投影到离散像素上，这一步可以看作是一种特殊的栅格化过程。

### What is Splatting (泼贱)？
泼贱是一种渲染技术，形象的说是将3D点或体素"泼洒"到2D图像平面的技术。泼贱技术的关键：
- 1.核函数：每个3D点或者体素都被视为一个局部影响区域，用一个核函数表示，常见的核函数包括：高斯核，EWA（椭圆加权平均）等
- 2.投影过程：需要将核函数(包括影响范围和强度)投影到图像平面，而不是简单的投影单个点
- 3.混合(Blending)：多个点对于同一个像素的贡献要正确的混合

上述的关键点出现在论文中的每个核心创新点中。

### Multivariate Gaussian Distribution (多元高斯分布)
$N$维随机变量$\mathbf{X}=[x_1, x_2, ..., x_N]$服从多元高斯，计作$\mathbf{X} \sim \mathcal{N}(\mathbf{\mu}, \mathbf{\Sigma})$，则它的概率密度函数为：

$$p(x) = \frac{1}{(2\pi)^{N/2} |\mathbf{\Sigma}|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{x} - \mathbf{\mu})^\top \mathbf{\Sigma}^{-1} (\mathbf{x} - \mathbf{\mu}) \right)$$

其中：$\mathbf{\mu} \in \mathbb{R}^{N}$是均值向量，$\mathbf{\Sigma} \in \mathbb{R}^{N \times N}$是协方差矩阵，协方差矩阵是**半正定对称矩阵**。



## Intro

