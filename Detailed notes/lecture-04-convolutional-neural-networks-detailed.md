# Lecture 04：使用卷积神经网络进行图像分类
## Image Classification with Convolutional Neural Networks ｜结构化知识详稿

> 整理自 CS231n 第 4 课视频字幕、课件，以及学习过程中的连续追问。本文先用一次完整 CNN 训练闭环串起全部知识，再拆解关键概念。

---

## 0. 一次完整 CNN 训练闭环

### 一句话核心

CNN 的所有卷积核和全连接权重在模型建立时一次性初始化；前向传播中，各层处理上一层产生的特征图并得到 loss；反向传播中，loss 沿原路径反向计算所有参数的梯度，优化器统一更新它们。

### 示例网络

```text
输入 3×32×32
→ Conv1：6 个 3×5×5 卷积核，stride=1，padding=0
→ ReLU
→ MaxPool：2×2，stride=2
→ Conv2：10 个 6×5×5 卷积核，stride=1，padding=0
→ ReLU
→ MaxPool：2×2，stride=2
→ Flatten
→ Fully Connected：250 → 10
→ Softmax Cross-Entropy Loss
```

模型建立时一次性初始化：

$$
W_1\in\mathbb{R}^{6	imes3	imes5	imes5},\quad
W_2\in\mathbb{R}^{10	imes6	imes5	imes5},\quad
W_3\in\mathbb{R}^{10	imes250}
$$

前向传播：

$$
3	imes32	imes32

ightarrow6	imes28	imes28

ightarrow6	imes14	imes14

ightarrow10	imes10	imes10

ightarrow10	imes5	imes5

ightarrow250

ightarrow10
$$

$$
Z_1=W_1*X+b_1,
\quad A_1=\operatorname{ReLU}(Z_1),
\quad P_1=\operatorname{Pool}(A_1)
$$

$$
Z_2=W_2*P_1+b_2,
\quad A_2=\operatorname{ReLU}(Z_2),
\quad P_2=\operatorname{Pool}(A_2)
$$

$$
h=\operatorname{Flatten}(P_2),
\quad s=W_3h+b_3
$$

$$
p_j=rac{e^{s_j}}{\sum_k e^{s_k}},
\qquad L_i=-\log p_y
$$

反向传播：

```text
L → logits → W3 / h → P2 → Pool2 → ReLU2 → W2 / P1
  → Pool1 → ReLU1 → W1
```

得到：

$$
rac{\partial L}{\partial W_3},\quad
rac{\partial L}{\partial W_2},\quad
rac{\partial L}{\partial W_1}
$$

下一批继续使用更新后的参数，而不是重新初始化。

### 关键对象区分

| 对象 | 本质 | 生命周期 |
|---|---|---|
| 卷积核 / 权重 | 模型参数 | 初始化一次，训练中持续更新和保留 |
| 偏置 | 模型参数 | 初始化一次，训练中持续更新和保留 |
| 特征图 | 当前输入的中间数据 | 每次 forward 重新计算 |
| 梯度 | loss 对变量的敏感度 | 每次 backward 重新计算 |
| 优化器状态 | 动量、一阶/二阶矩 | 跨训练步骤保存 |

---

## 1. 从全连接网络到 CNN

全连接层将图像展平：

$$
X\in\mathbb{R}^{3	imes H	imes W}

ightarrow
x\in\mathbb{R}^{3HW}
$$

然后：

$$
h=\phi(Wx+b)
$$

数值未消失，但模型结构没有显式利用二维邻接关系和位置共享。CNN 的核心改造是：保留空间结构，只让神经元读取局部区域，并在所有位置复用同一检测器。

---

## 2. 卷积层：局部模板匹配

某个位置：

$$
y_{k,i,j}
=
\sum_{c,u,v}W_{k,c,u,v}X_{c,i+u,j+v}+b_k
$$

卷积核是可学习参数，输入图像块是数据。二者逐元素相乘求和，得到一个局部匹配响应。

一个卷积核在所有空间位置计算后产生一张特征图；多个卷积核产生多张特征图。

---

## 3. 参数共享与多个卷积核

同一卷积核跨位置共享参数：

```text
左上角使用 K1
中央使用 K1
右下角仍使用 K1
```

不同卷积核之间则通常不同：

```text
K1：可能偏向竖直边缘
K2：可能偏向水平边缘
K3：可能偏向颜色对比
```

多个卷积核可以读取同一片输入数据，从不同角度描述同一区域。

---

## 4. 初始化与特征学习

卷积核通常使用 Xavier、He 等受控随机初始化。随机初始化的作用是打破对称性，而不是规定某个卷积核必须学成某种特征。

若参数和后续结构完全对称：

```text
相同参数 → 相同输出 → 相同梯度 → 相同更新
```

最终特征由初始化、数据、任务、网络结构、损失和优化共同决定。同一初始化在不同数据集上完全可能分别学成竖直边缘、水平边缘或其他模式。

---

## 5. 卷积核尺寸、内积与输出尺寸

方正的是观察窗口，不是特征本身。$3	imes3$ 窗口内部可以学习水平、竖直、斜向或不规则局部模式。

输出尺寸：

$$
H_{	ext{out}}=
\left\lfloorrac{H-K_H+2P_H}{S_H}
ight
floor+1
$$

$$
W_{	ext{out}}=
\left\lfloorrac{W-K_W+2P_W}{S_W}
ight
floor+1
$$

同尺寸填充（stride=1，奇数核）：

$$
P_H=rac{K_H-1}{2},\qquad P_W=rac{K_W-1}{2}
$$

卷积核数量决定输出通道，卷积核高宽 / stride / padding 决定输出高宽。

---

## 6. ReLU、池化与下采样

卷积是线性的，所以层间需要 ReLU：

$$
\operatorname{ReLU}(x)=\max(0,x)
$$

池化把局部区域的多个数汇总成一个代表值：

$$
y=\max(x_1,\dots,x_n)
$$

或：

$$
y=rac1n\sum_i x_i
$$

池化输出尺寸：

$$
H_{	ext{out}}=
\left\lfloorrac{H-K_H}{S_H}
ight
floor+1
$$

$$
W_{	ext{out}}=
\left\lfloorrac{W-K_W}{S_W}
ight
floor+1
$$

下采样不是优化器，而是空间压缩：减少计算、扩大有效感受野，并牺牲部分精细位置。

---

## 7. 多层卷积如何协同

第一层：

```text
原始像素 → 边缘 / 颜色 / 简单纹理
```

第二层：

```text
第一层多个通道 → 曲线 / 角点 / 更复杂纹理
```

更深层：

```text
局部结构 → 物体部件 → 语义特征
```

所有层从训练开始就共同学习，不是先训练完第一层再创建第二层。

---

## 8. 全连接、Softmax 与 loss

$$
P_2\in\mathbb{R}^{10	imes5	imes5}

ightarrow
h\in\mathbb{R}^{250}
$$

$$
s=W_3h+b_3\in\mathbb{R}^{10}
$$

$$
p_j=rac{e^{s_j}}{\sum_k e^{s_k}}
$$

$$
L_i=-\log p_y
$$

卷积层负责提供特征，全连接层负责综合特征形成类别证据，loss 统一评价整个系统。

---

## 9. 反向传播与 AdamW

反向传播负责：

$$

abla_{W_1}L,\quad
abla_{W_2}L,\quad
abla_{W_3}L
$$

优化器负责更新：

$$
W\leftarrow W-\eta
abla_WL
$$

AdamW 使用梯度的一阶、二阶移动平均，并解耦权重衰减。但它不替代 forward、loss 或 backward，只替代最后的参数更新规则。

---

## 10. 本讲最重要的闭环

```text
卷积核 = 持久存在的可学习参数
特征图 = 针对当前输入临时生成的数据

所有层参数一次性初始化
→ 每层使用自己的参数做 forward
→ 最终得到 logits 和 loss
→ backward 为所有层计算梯度
→ optimizer 更新所有参数
→ 下一批继续使用更新后的参数
```

---

## 11. 常见误区

1. 卷积核是从图片中切出来的。——卷积核是参数，图像块是数据。
2. 第二层运行到时才初始化。——所有层在模型创建时就已初始化。
3. 第一层做完后被丢弃。——它每次处理原图，其输出供后层使用，并持续被 loss 更新。
4. 池化产生新卷积核。——池化产生下采样后的特征图。
5. 参数共享表示所有卷积核一样。——共享发生在同一核的不同位置。
6. 随机初始化保证不同特征。——只打破对称性，不保证严格分工。
7. 下采样是优化参数。——下采样压缩空间，优化器才更新参数。
8. Softmax 就是 loss。——Softmax 生成概率，交叉熵生成 loss。
9. AdamW 替代反向传播。——backward 算梯度，AdamW 用梯度更新参数。

---

## 12. 自测问题

1. 为什么 $3	imes5	imes5$ 卷积核在一个位置只输出一个数？
2. 为什么 6 个卷积核输出 6 张特征图，而不是把原图切成 6 块？
3. 为什么第二层卷积核必须覆盖前一层全部 6 个通道？
4. 参数共享与不同卷积核完全相同有什么区别？
5. 为什么池化通常不改变通道数？
6. 为什么所有层可以同时从随机状态开始训练？
7. Softmax 和交叉熵各自负责什么？
8. `loss.backward()` 与 `optimizer.step()` 分别做什么？

---

## 一句话收束

卷积神经网络不是一层层临时制造卷积核，而是用一组从训练开始就存在的参数化操作，把原始像素逐层转换为更高级特征；最终分类损失再沿整个计算图反向更新所有层，使特征提取和类别判断共同学成。
