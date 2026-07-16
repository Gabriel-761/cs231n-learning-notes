# CS231n 学习笔记 | 一个学习者的深度学习视觉之路

> 这是我系统学习斯坦福 CS231n（计算机视觉与深度学习）的完整笔记。  
> 与网上的知识点复述不同，我刻意记录了**自己真实卡住的地方和想通的过程**——  
> 因为对初学者最有用的，往往不是标准答案，而是看到另一个人怎么趟过同一个坑。

---

## 📌 这份笔记的特点

- **带思考痕迹**：每讲都有「我的疑问与突破」栏，记录真实的理解过程。
- **重串联**：不只罗列知识点，而是用一条逻辑链把它们前因后果串起来。
- **可自测**：每讲附自测题与解答，学完即可检验。
- **持续更新**：按我自己的节奏，学一讲、整理一讲。
- **双层笔记**：`notes/` 保存我二次消化后的正式 Markdown 学习笔记；`Detailed notes/` 保存 GPT 生成的 PDF 结构化详稿，便于回查完整解释。

---

## 📚 目录

| 讲次 | 主题 | 状态 |
|------|------|------|
| [Lecture 01](notes/lecture-01-knn-linear-classifier.md) | 图像分类：从 kNN 到线性分类器 | ✅ 已完成 |
| [Lecture 02](notes/lecture-02-regularization-optimization.md) | 正则化与优化：SGD / Momentum / RMSProp / Adam / AdamW | ✅ 已完成 |
| [Lecture 03](notes/lecture-03-neural-networks-backpropagation.md) | 神经网络与反向传播：MLP / 激活函数 / 计算图 / Backpropagation | ✅ 已完成 |
| [Lecture 04](notes/lecture-04-convolutional-neural-networks.md) | 卷积神经网络基础：卷积 / 参数共享 / 池化 / 完整训练闭环 | ✅ 已完成 |
| [Lecture 05](notes/lecture-05-cnn-architectures-training.md) | CNN 架构与训练：归一化 / Dropout / VGG / ResNet / 权重衰减 | ✅ 已完成 |

> Lecture 03 的完整结构化详稿见：[`Detailed notes/lecture-03-neural-networks-backpropagation.md`](Detailed%20notes/lecture-03-neural-networks-backpropagation.md)
>
> Lecture 04 的完整结构化详稿见：[`Detailed notes/lecture-04-convolutional-neural-networks.md`](Detailed%20notes/lecture-04-convolutional-neural-networks.md)
>
> Lecture 05 的完整结构化详稿见：[`Detailed notes/lecture-05-cnn-architectures-training-detailed.pdf`](Detailed%20notes/lecture-05-cnn-architectures-training-detailed.pdf)

---

## 🗂️ 仓库结构

```
notes/            每讲一份正式学习笔记，强调“我如何理解”
Detailed notes/   GPT 生成的 PDF 结构化详稿，保留完整解释、类比、公式和自测
ppt/              课程原始 PPT / 课件（本地留存，不建议提交到公开仓库，见下方说明）
assets/           图片、手绘、截图
exercises/        自测题与解答
resources.md      课程链接与参考资料
```

> ⚠️ **关于 `ppt/` 文件夹**：斯坦福 CS231n 的课件本身属于课程版权内容，个人学习笔记（notes/ 下的 md 文件）可以公开分享，但建议不要把原始 PPT/课件文件提交到公开仓库。可以在项目根目录加一行 `ppt/` 到 `.gitignore`，只在本地保留用于个人复习。

---

## 🎯 关于我

正在系统学习计算机视觉与深度学习。这个仓库既是我的学习轨迹，
也希望能给后来的学习者一点参考。欢迎 Issue 交流或指出错误。

---

*本笔记基于斯坦福 CS231n 公开课程整理，仅供学习交流使用。*
