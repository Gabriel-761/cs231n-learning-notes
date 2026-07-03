# Lecture 03：神经网络与反向传播

> **课程**：CS231n · Lecture 4 Neural Networks and Backpropagation（仓库编号 Lecture 03）  
> **学习日期**：2026-07-03  
> **一句话主线**：上一讲知道了要用梯度下降更新参数，这一讲解决“模型变成多层神经网络后，梯度到底怎么系统、高效地算出来”。

---

## 0. 本讲脉络

```
线性分类器 f = Wx + b
  ↓ 表达能力有限，只能做线性分割
加入隐藏层与激活函数
  ↓ 多层线性变换 + 非线性，得到更强表达能力
神经网络 forward pass
  ↓ 得到类别分数与 loss
问题：复杂网络中 ∂L/∂W 怎么算？
  ↓ 手推导太繁琐，换 loss / 换结构就要重来
计算图 Computational Graph
  ↓ 把复杂函数拆成简单节点
反向传播 Backpropagation
  ↓ 沿计算图反向应用链式法则
得到所有参数梯度
  ↓ 交给 SGD / Adam 等优化器更新
进入后续 CNN / Transformer 等复杂网络
```

---

## 1. 核心知识点

### 1.1 从线性分类器到神经网络

**一句话**：神经网络是在多个线性变换之间插入非线性激活函数，从而比线性分类器拥有更强的表达能力。

**展开**：

线性分类器：

$$
f = Wx + b
$$

两层全连接神经网络（Fully Connected Network / MLP）：

$$
h = \phi(W_1x + b_1)
$$

$$
s = W_2h + b_2
$$

合起来：

$$
s = W_2\phi(W_1x+b_1)+b_2
$$

其中：

- $x$：输入，比如一张图像展平成的向量。
- $W_1,b_1$：输入层到隐藏层的权重和偏置。
- $h$：隐藏层输出，也叫隐藏表示（hidden representation）。
- $\phi$：激活函数（activation function），例如 ReLU。
- $W_2,b_2$：隐藏层到输出层的权重和偏置。
- $s$：类别分数（class scores）。

**直觉/类比**：

线性分类器是“直接看像不像某一类”；神经网络是“先提取边缘、纹理、局部形状等中间特征，再组合成类别判断”。

```
线性分类器：像素 → 类别
神经网络：像素 → 边缘/纹理 → 局部结构 → 类别
```

---

### 1.2 神经元、层、网络的关系

**一句话**：神经元是一个“加权求和 + 激活”的计算单元；多个神经元并排组成一层，多层串联组成网络。

**展开**：

单个神经元：

$$
z = w^Tx + b
$$

$$
h = \phi(z)
$$

一层有 $H$ 个神经元时，可以用矩阵统一写成：

$$
h = \phi(Wx+b)
$$

这里：

- $W$ 的每一行对应一个神经元的权重。
- $b$ 是 bias vector，每个神经元通常有一个自己的偏置。
- $\phi$ 对每个神经元输出逐元素作用。

**直觉/类比**：

- 神经元是“特征检测器”。
- 一层是一排检测器。
- 多层网络是把前一层检测出的特征继续组合，形成更高级特征。

注意：神经元不完全等于线性分类器。更准确地说，一个神经元内部有类似线性打分的部分 $w^Tx+b$，但它通常只是输出一个中间特征；线性分类器可以看作一层输出神经元组成的最简单网络。

---

### 1.3 为什么一定要有非线性激活函数

**一句话**：没有非线性激活，多层线性变换合起来仍然只是一个线性变换。

**展开**：

如果去掉激活函数：

$$
s = W_2W_1x
$$

令：

$$
W_3 = W_2W_1
$$

则：

$$
s = W_3x
$$

这又退化成了线性分类器。

常见激活函数：

- ReLU：$\mathrm{ReLU}(x)=\max(0,x)$
- Leaky ReLU：负半轴保留一个小斜率，缓解 dead neuron。
- Sigmoid：$\sigma(x)=\frac{1}{1+e^{-x}}$，常用于二分类输出层。
- Tanh：输出范围为 $(-1,1)$。
- GELU / SiLU：常见于 Transformer 或现代 CNN。

**直觉/类比**：

只叠线性层就像连续做拉伸、旋转，最后仍可合并成一次大变换；加入非线性后，模型才能“弯折空间”，把原本线性不可分的数据变得可分。

---

### 1.4 模型容量与正则化

**一句话**：隐藏层神经元越多，模型能表示的函数越复杂；但容量太大容易把训练集噪声也学进去，所以通常保留足够大的网络，再用正则化控制过拟合。

**展开**：

模型容量（model capacity）可以理解为模型“能画出多复杂规则”的能力。

- 容量太小：学不会真实复杂规律，欠拟合（underfitting）。
- 容量太大且无约束：可能记住训练集噪声，过拟合（overfitting）。
- 合适做法：让模型有足够表达能力，再用正则化约束它不要乱记。

上一讲总损失：

$$
L(W)=\frac{1}{N}\sum_i L_i + \lambda R(W)
$$

这一讲可以理解为：神经网络提供表达能力，正则化控制自由度。

**直觉/类比**：

```
小网络：脑容量不够，复杂题学不会
大网络 + 无正则：什么都背，包括噪声和错题
大网络 + 合适正则：有能力理解复杂规律，但不鼓励死记硬背
```

---

### 1.5 Bias 是什么，怎么得到

**一句话**：bias 是每个神经元 / 每一层的可学习偏移量，和权重一样通过反向传播与优化器学出来。

**展开**：

单个神经元：

$$
z=w^Tx+b
$$

$w$ 决定输入特征的重要性，$b$ 决定整体偏移。没有 $b$ 时，决策边界会受到更多限制；有 $b$ 后，边界可以平移。

训练过程：

```
初始化 W 和 b
  ↓
forward 计算预测
  ↓
计算 loss
  ↓
backward 计算 ∂L/∂W 和 ∂L/∂b
  ↓
优化器更新 W 和 b
```

更新公式：

$$
b \leftarrow b - \alpha\frac{\partial L}{\partial b}
$$

**直觉/类比**：

$w$ 像“判断方向”，$b$ 像“判断门槛”。比如 ReLU 神经元什么时候被激活，不只取决于输入方向，也取决于门槛位置。

---

### 1.6 Forward pass 与计算图

**一句话**：前向传播负责从输入算到输出和 loss；计算图把这个复杂过程拆成可复用的小操作节点。

**展开**：

前向传播：

```
x → W1x+b1 → activation → h → W2h+b2 → scores → loss
```

计算图（computational graph）把复杂函数拆成许多节点：

- 矩阵乘法
- 加法
- ReLU / sigmoid
- loss
- regularization

每个节点只需要实现：

```python
forward()   # 算输出，并缓存 backward 需要的中间值
backward()  # 接收上游梯度，计算下游梯度
```

**直觉/类比**：

计算图像一张生产流程图。forward 是生产产品，backward 是质检后沿流程追责：每个工序只需要知道自己接收了什么、做了什么、该把责任往前分多少。

---

### 1.7 反向传播：链式法则的模块化应用

**一句话**：反向传播就是沿计算图反向递归使用链式法则，计算 loss 对所有参数和中间变量的梯度。

**展开**：

简单例子：

$$
f=(x+y)z
$$

令：

$$
q=x+y
$$

则：

$$
f=qz
$$

局部导数：

$$
\frac{\partial q}{\partial x}=1,\quad \frac{\partial q}{\partial y}=1
$$

$$
\frac{\partial f}{\partial q}=z,\quad \frac{\partial f}{\partial z}=q
$$

链式法则：

$$
\frac{\partial f}{\partial x}=\frac{\partial f}{\partial q}\frac{\partial q}{\partial x}=z
$$

$$
\frac{\partial f}{\partial y}=\frac{\partial f}{\partial q}\frac{\partial q}{\partial y}=z
$$

三个术语：

- 上游梯度（upstream gradient）：后面传来的梯度。
- 局部梯度（local gradient）：当前节点自己的导数。
- 下游梯度（downstream gradient）：当前节点传给前面节点的梯度。

核心规则：

$$
\text{downstream} = \text{upstream} \times \text{local gradient}
$$

---

### 1.8 Jacobian：向量函数的导数表

**一句话**：Jacobian 是向量到向量函数的局部梯度，用来记录“每个输入分量对每个输出分量的影响”。

**展开**：

若：

$$
z=f(x),\quad x\in\mathbb{R}^n,\quad z\in\mathbb{R}^m
$$

则 Jacobian 为：

$$
J=\frac{\partial z}{\partial x}
=
\begin{bmatrix}
\frac{\partial z_1}{\partial x_1} & \cdots & \frac{\partial z_1}{\partial x_n}\\
\vdots & \ddots & \vdots\\
\frac{\partial z_m}{\partial x_1} & \cdots & \frac{\partial z_m}{\partial x_n}
\end{bmatrix}
$$

反向传播中：

$$
\frac{\partial L}{\partial x}=J^T\frac{\partial L}{\partial z}
$$

但实际训练中通常不显式构造 Jacobian，因为它可能非常巨大。我们只计算 Jacobian 乘以上游梯度的结果。

**直觉/类比**：

Jacobian 是一张“影响关系表”：每一格都回答“某个输入元素变一点，某个输出元素变多少”。

---

### 1.9 矩阵乘法的反向传播

**一句话**：矩阵乘法的 backward 不需要显式 Jacobian，只要用形状匹配的矩阵乘法公式即可。

**展开**：

设：

$$
Y=XW
$$

其中：

$$
X\in\mathbb{R}^{N\times D},\quad W\in\mathbb{R}^{D\times M},\quad Y\in\mathbb{R}^{N\times M}
$$

给定：

$$
dY=\frac{\partial L}{\partial Y}
$$

则：

$$
dX=dYW^T
$$

$$
dW=X^TdY
$$

形状检查：

$$
dX: (N\times M)(M\times D)=N\times D
$$

$$
dW: (D\times N)(N\times M)=D\times M
$$

**直觉/类比**：

标量乘法 backward 是“乘以另一个输入”；矩阵乘法也是类似，只是为了让形状对齐，需要转置。

---

## 2. 我的疑问与突破 ⭐

**疑问 1**：神经元是不是线性分类器？网络和神经元是什么关系？  
**怎么想通的**：神经元不是完整线性分类器，而是“线性打分 + 激活”的小单元。多个神经元并排构成一层；上一层输出作为下一层输入，多层串联就是网络。线性分类器可以看作最简单的一层输出网络。

**疑问 2**：bias 是什么？它是结果吗，怎么得到？  
**怎么想通的**：bias 不是结果，而是和权重一样的可学习参数。它决定神经元的整体偏移或激活门槛，通过 $\partial L/\partial b$ 反向传播得到梯度，再由优化器更新。

**疑问 3**：模型容量为什么和神经元数量有关？为什么不用网络大小当 regularizer？  
**怎么想通的**：神经元越多，能组合出的中间特征和决策边界越复杂，容量越大。但把网络做小可能让模型连真实复杂规律都学不会，所以更常见策略是保留足够容量，再用正则化控制它不要死记训练噪声。

**疑问 4**：Jacobian 矩阵为什么要提出？  
**怎么想通的**：普通导数只够描述标量到标量，梯度描述向量到标量；而神经网络的中间层常常是向量到向量，所以需要 Jacobian 记录所有“输入分量 → 输出分量”的局部影响。但工程中不显式构造它，只隐式计算它与上游梯度相乘的结果。

---

## 3. 易混淆点辨析

| 概念 A | 概念 B | 关键区别 |
|--------|--------|----------|
| 神经元 | 线性分类器 | 神经元通常输出一个中间特征；线性分类器输出多个类别分数。 |
| 权重 $W$ | 偏置 $b$ | $W$ 控制输入特征方向与重要性；$b$ 控制整体偏移 / 激活门槛。 |
| 层 | 网络 | 一层是多个神经元并排；网络是多层串联。 |
| 线性变换 | 激活函数 | 线性变换做加权组合；激活函数引入非线性。 |
| forward | backward | forward 从输入算到 loss；backward 从 loss 反向算梯度。 |
| local gradient | upstream gradient | local 是当前节点自己的导数；upstream 是后面传来的梯度。 |
| gradient | Jacobian | gradient 常用于向量到标量；Jacobian 用于向量到向量。 |
| 理论 Jacobian | 实际 backward | 理论上有 Jacobian；实际中通常用更高效的隐式公式。 |

---

## 4. 一页速记

- 神经网络：

$$
h=\phi(W_1x+b_1),\quad s=W_2h+b_2
$$

- 没有激活函数：

$$
W_2W_1x=W_3x
$$

多层线性仍是线性。

- ReLU：

$$
\mathrm{ReLU}(x)=\max(0,x)
$$

- 训练链条：

```
forward 得到 loss → backward 得到梯度 → optimizer 更新参数
```

- 反向传播核心：

$$
\text{downstream} = \text{upstream}\times\text{local gradient}
$$

- 矩阵乘法反传：

$$
Y=XW
$$

$$
dX=dYW^T,\quad dW=X^TdY
$$

- 最关键结论：

```
神经网络 = 多层线性变换 + 非线性激活
反向传播 = 计算图上的链式法则
Jacobian = 向量函数的局部导数表，但实际训练中隐式使用
```

---

## 5. 输出检验：自测题

**【简单】** 为什么没有激活函数时，多层网络仍然只是线性分类器？

<details><summary>参考答案</summary>

因为多个线性变换复合仍是线性变换，例如 $W_2W_1x$ 可以合并成 $W_3x$。所以没有非线性激活时，增加层数不会提升本质表达能力。

</details>

**【中等】** 一个神经元中的 $b$ 是什么？它如何在训练中更新？

<details><summary>参考答案</summary>

$b$ 是 bias，表示可学习偏移量。它和权重一样参与 forward：$z=w^Tx+b$。反向传播计算 $\partial L/\partial b$，优化器用 $b\leftarrow b-\alpha\partial L/\partial b$ 更新它。

</details>

**【中等】** 设 $Y=XW$，已知 $dY=\partial L/\partial Y$，写出 $dX$ 和 $dW$。

<details><summary>参考答案</summary>

$$
dX=dYW^T
$$

$$
dW=X^TdY
$$

可以用形状检查记忆：$dX$ 必须和 $X$ 同形状，$dW$ 必须和 $W$ 同形状。

</details>

**【进阶】** 为什么理论上有 Jacobian，但实际深度学习框架不显式构造它？

<details><summary>参考答案</summary>

Jacobian 记录向量到向量函数的所有偏导关系，但在神经网络中可能非常巨大，占用大量内存。实际框架只计算 Jacobian 与上游梯度相乘后的结果，也就是为每个操作写高效 backward 公式。

</details>

---

## 6. 延伸 & 待办

- [ ] 手写一个两层 MLP 的 forward / backward，验证 $dW_1,dW_2,db_1,db_2$ 的形状。
- [ ] 用 NumPy 实现 ReLU backward：`dx = dout * (x > 0)`。
- [ ] 对矩阵乘法 $Y=XW$ 做一次数值梯度检查，确认 $dX,dW$ 正确。
- [ ] 下一讲重点预习：卷积神经网络 CNN 中“局部连接”和“参数共享”如何改变全连接网络。

参考资料：CS231n Lecture 4: Neural Networks and Backpropagation；CS231n course notes: Backpropagation and Neural Networks。
