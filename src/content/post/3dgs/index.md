---
title: "3D Gaussian Splatting notes"
description: "3dgs learning notes"
publishDate: "3 Dec 2024"
updatedDate: "18 Jan 2025"
coverImage:
  src: "./figs/3dgs_pipeline.svg"
  alt: "3DGS pipeline"
tags: ["tech/simulation"]
draft: false
---

> 3D Gaussian Splatting(3DGS)由2023年论文[3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079)中提出，由于渲染速度快且质量高，迅速超越(NeRF)称为新视角渲染的热点，涌现出大量优秀的后续工作。本篇笔记以论文原文和原作者的[教程](https://3dgstutorial.github.io/3dv_part1.pdf)为基础，包括简单的渲染代码实现，训练时关键参数的初始化等，不包括模型的训练代码。

## Background
如果不了解图形学以及忘记一些基础知识，这里进行简单的介绍。

### Rasterization
栅格化（Rasterization）是将矢量图形转换为像素点阵的过程，是计算机图形学中的一个基础概念。简单来说，就是把连续的几何图形（如线段、多边形、曲线等）转换成离散的像素点来显示在屏幕上。Gaussian Splatting在最终显示时仍需要将结果投影到离散像素上，这一步可以看作是一种特殊的栅格化过程。

### Splatting
泼贱（Splatting）是一种渲染技术，形象的说是将3D点或体素"泼洒"到2D图像平面的技术。泼贱技术的关键：
- 核函数：每个3D点或者体素都被视为一个局部影响区域，用一个核函数表示，常见的核函数包括：高斯核，EWA（椭圆加权平均）等
- 投影过程：需要将核函数(包括影响范围和强度)投影到图像平面，而不是简单的投影单个点
- 混合(Blending)：多个点对于同一个像素的贡献要正确的混合，可以按照深度对所有的核函数进行排序

### Multivariate Gaussian Distribution
$N$维随机变量$\mathbf{x}=[x_1, x_2, ..., x_N]$服从多元高斯分布，记作$\mathbf{x} \sim \mathcal{N}(\mathbf{\mu}, \mathbf{\Sigma})$，则它的概率密度函数为：

$$
\begin{equation}
  p(\mathbf{x}) = \frac{1}{(2\pi)^{N/2} |\mathbf{\Sigma}|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{x} - \mathbf{\mu})^\top \mathbf{\Sigma}^{-1} (\mathbf{x} - \mathbf{\mu}) \right)
\end{equation}
$$

其中：$\mathbf{\mu} \in \mathbb{R}^{N}$是均值向量，$\mathbf{\Sigma} \in \mathbb{R}^{N \times N}$是协方差矩阵，$|\mathbf{\Sigma}|$是协方差矩阵行列式值，$\mathbf{\Sigma}^{-1}$为协方差矩阵的逆。
$(\mathbf{x} - \mathbf{\mu})^\top \mathbf{\Sigma}^{-1} (\mathbf{x} - \mathbf{\mu})$称为马氏距离，用于衡量一个点$\mathbb{x}$与一个分布之间的距离，是一个**标量值**。

随机变量$x$和$y$的协方差的计算：$\Sigma(x,y)=E[(x-\mu_{x})(y-\mu_{y})]$，描述两个随机变量之间的相关性，如果协方差为正，则表示两个变量正相关，即同方向变化，如果为负，则负相关，如果为零，则不相关；协方差值的大小没有比较的意义。协方差矩阵就是多个随机变量协方差构成的方阵。

:::note
多元随机变量$\mathbf{x}$的协方差矩阵是$\Sigma$，那么$\mathbf{x}$经过线性变换$\mathbf{A}$后的协方差矩阵是？先说结论： $\Sigma(A\mathbf{x})=A\Sigma(\mathbf{x})A^\top$

我们来推导一下。令$\mathbf{y}=\mathbf{Ax}$，则$\Sigma(\mathbf{y}) = E[(\mathbf{y} - E[\mathbf{y}])(\mathbf{y} - E[\mathbf{y}])^\top]$，由于$E[\mathbf{y}] = E[\mathbf{Ax}] = \mathbf{A}E[\mathbf{x}]$，代入可知$\Sigma(\mathbf{y})=E[(\mathbf{Ax}-\mathbf{A}E[\mathbf{x}])(\mathbf{Ax}-\mathbf{A}E[\mathbf{x}])^\top]$。

由于$\mathbf{A}$是一个常数矩阵，将上面式子中的$\mathbf{A}$提取出来，$\Sigma(\mathbf{y})=E[\mathbf{A}(\mathbf{x}-E[\mathbf{x}])(\mathbf{x}-E[\mathbf{x}])^\top\mathbf{A}^{\top}]=\mathbf{A}\Sigma(\mathbf{x})\mathbf{A}^\top$，证明完毕。
:::

协方差矩阵是**半正定对称矩阵**，即:

$$
\begin{align}
  \Sigma=\Sigma^\top \rightarrow cov(x_i, x_j)&=cov(x_j,x_i) \\
  \mathbf{z}^\top \mathbf{\Sigma} \mathbf{z} \geq 0 ,\quad \forall \mathbf{z} \in \mathbb{R}^N
\end{align}
$$

**协方差矩阵的几何意义**：协方差矩阵的主对角线元素是方差，表示的是分布在特征轴上的离散程度，可以形象的理解是分布的高矮胖瘦；非主对角线元素是协方差，表示的是分布的方向(orientation)或者说旋转。下图[^1]是一个二维多元高斯分布，主对角线元素决定了分布的大小，而非主对角线元素决定了分布的旋转。三维高斯分布是一个椭球。

![covariance](./figs/cov.png)

[^1]: [协方差矩阵的几何解释](https://njuferret.github.io/2019/07/28/2019-07-28_geometric-interpretation-covariance-matrix/)

### Quadratic Form
多元高斯分布的定义中使用了二次型（Quadratic Form），论文中也出现多次二次型形式的矩阵相乘，这里简单介绍。

对于一个对称矩阵$\mathbf{A} \in \mathbb{R}^{N \times N}$和一个**列**向量$\mathbf{z} \in \mathbb{R}^{N}$，二次型的定义如下，二次型的结果为一个**标量**。

$$
Q(\mathbf{z})=\mathbf{z}^\top \mathbf{A} \mathbf{z}
$$

如果见到$\mathbf{z}A\mathbf{z}^\top$，它是什么？有两种可能：1）$\mathbf{z}$是行向量，那么其实和$\mathbf{z}^\top \mathbf{A} \mathbf{z}$是等价的；2）$\mathbf{z}$可能是一个矩阵，那么上面式子只是矩阵相乘了。

### Jacobian Matrix
假设某**向量值函数** $\mathbf{f}: \mathbb{R}^n \rightarrow \mathbb{R}^m$，即从$\mathbf{x} \in \mathbb{R}^n$映射到向量$\mathbf{f(x)}\in \mathbb{R}^m$，其雅可比矩阵是$m\times n$的矩阵

$$
J = \left[ \frac{\partial f}{\partial x_1} \cdots \frac{\partial f}{\partial x_n} \right] = \left[ \begin{array}{cccc}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{array} \right]
$$

**雅可比矩阵描述了向量值函数在某一点附近的局部线性变换**，通俗的说，雅可比矩阵在某可微点的邻域范围内提供了向量值函数的近似**线性**表示(一阶泰勒展开)，可视化理解几何意义参考[这个视频](https://www.youtube.com/watch?v=bohL918kXQk)。在3D Gaussian投影过程中会遇到这个知识点。

雅可比矩阵与一阶泰勒展开。可微分函数$\mathbf{f(x)}$，其在某点$\mathbf{z}$的一阶泰勒展开表示为
$$
\mathbf{f(x)} \approx \mathbf{f(z)} + J(\mathbf{z})(\mathbf{x-z})
$$
其中，$\mathbf{f(z)}$是函数在$\mathbf{z}$处的值，$J(\mathbf{z})是$\mathbf{z}处的雅可比矩阵，$\mathbf{x-z}表示$\mathbf{z}$的邻域内点$\mathbf{x}$的差。

另外，向量值函数 $\mathbf{f}(\mathbf{x})=(f_1(\mathbf{x}), f_2(\mathbf{x}), ..., f_m(\mathbf{x}))$是输出值为向量的函数，其输入可以是一个标量或者向量。
举一个例子说明，假设有一个粒子在三维空间中运动，其位置随时间$t$变化，我们可以用一个向量值函数来描述这个粒子的位置：
$$\mathbf{r}(t)=\langle f(t), g(t), h(t) \rangle$$
其中，$f(t), g(t), h(t)$分别是例子在$x$轴，$y$轴和$z$轴上的变化。则$\mathbf{r}(t)$是一个向量值函数，输入是时间$t$，输出是三维的例子位置。

## 3D Gaussian Representation

三维空间有很多形式，例如显式的栅格化Voxel，或者隐式的Neural Radiance。3D Gaussian也是一种对三维空间的表征，用大量的3D Gaussians来更自由、更紧凑(相对于稠密、规则的Voxel)的表征三维空间。3D Gaussians的参数 ($\mathbf{\mu}$和$\mathbf{\Sigma}$) 构成了模型的**权重参数**之一，将三维场景的信息(通过训练)“压缩”到的模型参数中去(即3DGS模型的权重就是这些3D Gaussians参数)，可以用于新视角生成，也可以有更灵活的用途，甚至是自动驾驶的感知任务[^2]。3D Gaussians的表征也可以使用并行化实现高效的渲染。

[^2]: [GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction](https://arxiv.org/abs/2405.17429)

具体来说，3D Gaussian表征是一组定义在**世界坐标系下**的参数，包含：三维位置(3D position)，协方差(anisotropic covariance)，不透明度(opacity $\alpha$)和球谐函数(spherical harmonic, SH)：
- 3D位置是三维高斯分布的均值$\mathbf{\mu}$，有3个值
- 协方差是三维高斯分布的$\mathbf{\Sigma}$，可以拆分成主对角线元素3个值和表示三维旋转的四元数4个值，后面会更详细讲解
- 不透明度(opacity $\alpha$)，是一个标量值，用于$\alpha -$blending
- 球谐函数用来表示辐射神经场的带方向的颜色

## 3D Gaussian Rasterizer
3D Gaussians是定义在世界坐标系下的对三维空间的连续表征，需要进行[栅格化](#rasterization)渲染到离散空间的图片上。这里涉及到3D Gaussian的投影和渲染。

### 3D Covariance Splatting
3D Gaussian的位置(均值$\mathbf{\mu}$)投影到图像平面正常使用投影矩阵即可，但是3D Gaussians是三维空间中的椭球形状如何投影到图像平面，协方差矩阵决定了三维椭球的大小和方向，因此协方差矩阵的投影是关键，论文中使用如下公式近似投影协方差矩阵：
$$
\Sigma'=JW\Sigma W^{\top}J^{\top}
$$
其中，$W$是世界坐标系到相机坐标系的变换矩阵（transformation matrix），而$J$是投影变换(projective  transformation)的雅可比矩阵。

:::note
一般说投影(projective)矩阵，指的是三维空间中的点到二维图像平面的投影，它可以包括两部分：世界坐标系到相机坐标系变换，以及相机坐标系到图像平面的投影。如果只包含相机坐标系到图像坐标系的投影，也可以称作投影矩阵。
:::

:::note
我们简单推导下上面协方差投影矩阵的基本原理。假设存在一个线性变换$\mathbf{A}$，空间中的高斯分布$\mathbf{x} \sim \mathcal{N}(\mathbf{\mu}, \mathbf{\Sigma})$经过线性变换$\mathbf{A}$后，新的分布$\mathbf{y} \sim \mathcal{N}(\mathbf{A\mu}, \mathbf{A\Sigma A^\top})$，注意观察新的协方差矩阵与上面协方差投影公式的相似性。
:::

三维空间的点从世界坐标系到相机坐标系的变换矩阵是仿射变换，包含旋转矩阵(线性变换)和平移矩阵，因此可以直接使用针对线性变换的协方差矩阵投影。但是**透视投影变换是非线性的**，不能使用相机内参$\mathbf{K}$直接投影到相机平面，因此使用雅可比矩阵对投影变换进行线性近似。

雅可比矩阵提供某个可微点处的线性近似，哪具体是哪个点呢？实现中使用3D Gaussian的均值(mean)变换到相机坐标系后，进行雅可比矩阵的计算（更详细的推导需要阅读论文[^3]）：
```python
def compute_Jocobian(self, mean3d_N3):
    '''
    Compute the Jacobian of the affine approximation of the projective transformation.
    '''
    t = self.camera.world_to_camera(mean3d_N3)
    l = np.linalg.norm(t, axis=1, keepdims=True).flatten()
    # Compute the jacobian according to (29) from EWA Volume Splatting M.Zwicker et. al (2001)
    jacobian = np.zeros((t.shape[0], 3, 3))
    jacobian[:, 0, 0] = 1/t[:, 2]
    jacobian[:, 0, 2] = -t[:, 0]/t[:, 2]**2
    jacobian[:, 1, 1] = 1/t[:, 2]
    jacobian[:, 1, 2] = -t[:, 1]/t[:, 2]**2
    jacobian[:, 2, 0] = t[:, 0]/l
    jacobian[:, 2, 1] = t[:, 1]/l
    jacobian[:, 2, 2] = t[:, 2]/l

    return jacobian
```
[^3]: [Surface Splatting](https://dl.acm.org/doi/10.1145/383259.383300)

实际实现使用的是下面的公式：
$$
\Sigma_{2D}=JR\Sigma_{3D}R^{\top}J^{\top}
$$
其中，$R$是世界坐标系到相机坐标系面的旋转矩阵。

:::note
为什么从世界坐标系到图像平面的透视变换投影矩阵不是线性变换？投影矩阵如下：
$$
\begin{bmatrix} \mu \\ \nu \\ 1 \\ \end{bmatrix} = \frac{1}{Z_c}K \begin{bmatrix} X_c \\ Y_c \\ Z_c \\ \end{bmatrix}
$$
线性变换对**加法**和**数乘法封闭**，投影矩阵中引入了除法，使得**投影矩阵是非线性变换**。
:::

:::note
为什么投影矩阵实际中只有旋转部分没有平移部分？从[多元高斯分布经过线性变换后的协方差矩阵推导](#multivariate-gaussian-distribution)可知，平移对于协方差没有影响。
:::

### Covariance in practice
协方差矩阵的重要性质是对称矩阵和半正定，如果协方差矩阵作为模型参数进行梯度更新，那么很容易不满足协方差矩阵的性质。因此实际中，依靠协方差矩阵的几何意义，使用更直观的方式来近似表示协方差矩阵。

由于协方差矩阵的几何意义中对角线元素代表方差，即三维椭球大小（高矮胖瘦），非对角线元素代表旋转，因此将协方差矩阵拆分为表示大小的对角矩阵$S$和表示旋转的旋转矩阵$R$:
$$
\Sigma=RSS^\top R^\top
$$
实现中，使用一个$3$维向量$s$表示对角矩阵的元素，使用$4$维向量$q$表示四元数，即$7$个值表示协方差矩阵。

### alpha-blending

## Optimization

### 3D Gaussian Splatting Pipeline

### Adaptive Control

### 3D Gaussian Initailization

## Tile-based Rasterizer