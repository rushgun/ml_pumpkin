***降维***

****K近邻学习****
## 10.1(类别不一致的概率）

$$ P(e r r)=1-\sum_{c \in \mathcal{Y}} P(c | \boldsymbol{x}) P(c | \boldsymbol{z}) $$
公式意在表达某种错误率或不一致性的概率：

1. **错误或不一致性的概率 $P(err)$**：
   - 表示某个事件发生的错误率或概率，其中“err”可能代表错误（error）。

2. **条件概率 $P(c | \boldsymbol{x})$**：
   - 表示给定输入 $\boldsymbol{x}$ 的情况下，类别 $c$ 发生的概率。在机器学习中，$\boldsymbol{x}$ 可能代表特征向量。

3. **条件概率 $P(c | \boldsymbol{z})$**：
   - 表示给定另一组信息或特征 $\boldsymbol{z}$ 的情况下，类别 $c$ 发生的概率。这里 $\boldsymbol{z}$ 可能代表与 $\boldsymbol{x}$ 不同的另一组特征或数据表示。

4. **求和符号 $\sum_{c \in \mathcal{Y}}$**：
   - 表示对所有可能的类别 $c$ 进行求和，其中 $\mathcal{Y}$ 代表所有可能类别的集合。

5. **公式的含义**：
   - 公式 $P(err) = 1 - \sum_{c \in \mathcal{Y}} P(c | \boldsymbol{x}) P(c | \boldsymbol{z})$ 试图表达的是，在所有可能的类别上，给定两组不同信息 $\boldsymbol{x}$ 和 $\boldsymbol{z}$ 时，类别 $c$ 发生概率的乘积之和的补数。这意味着计算了某种形式的错误率或不一致性。

6. **可能的应用场景**：
   - 这个公式可能用于评估两组不同特征或数据源对于分类结果的一致性。如果 $P(c | \boldsymbol{x})$ 和 $P(c | \boldsymbol{z})$ 通常很接近，那么 $P(err)$ 会接近 0，表示高一致性；如果它们相差很大，则 $P(err)$ 会更高，表示低一致性。

7. **潜在问题**：
   - 公式中的条件概率通常不会直接相乘，除非在贝叶斯网络或某些特定的条件独立性假设下。此外，错误率的计算通常不会以这种方式表示，因此这个公式可能是特定领域的特定应用，或者是一个特定问题的简化模型。


## 10.2

$$
\begin{aligned}
P(e r r) &=1-\sum_{c \in \mathcal{Y}} P(c | \boldsymbol{x}) P(c | \boldsymbol{z}) \\
& \simeq 1-\sum_{c \in \mathcal{Y}} P^{2}(c | \boldsymbol{x}) \\
& \leqslant 1-P^{2}\left(c^{*} | \boldsymbol{x}\right) \\
&=\left(1+P\left(c^{*} | \boldsymbol{x}\right)\right)\left(1-P\left(c^{*} | \boldsymbol{x}\right)\right) \\
& \leqslant 2 \times\left(1-P\left(c^{*} | \boldsymbol{x}\right)\right)
\end{aligned}
$$
这个公式表达了在某种分类或预测任务中的错误率 $P(err)$ 的一个界限。让我们逐步分析这个公式：

1. **初始公式**：
   $$P(err) = 1 - \sum_{c \in \mathcal{Y}} P(c | \boldsymbol{x}) P(c | \boldsymbol{z})$$
   这表示错误率是1减去所有可能类别 $c$ 在给定两组数据 $\boldsymbol{x}$ 和 $\boldsymbol{z}$ 下的条件概率乘积的总和。这可能表示使用两组特征预测同一类别的概率一致性。

2. **简化假设**：
   $$P(err) \simeq 1 - \sum_{c \in \mathcal{Y}} P^{2}(c | \boldsymbol{x})$$
   这里假设 $P(c | \boldsymbol{x})$ 和 $P(c | \boldsymbol{z})$ 相等，或者在某种意义上可以互换，因此简化为 $P^2(c | \boldsymbol{x})$。

3. **进一步简化**：
   $$P(err) \leqslant 1 - P^{2}(c^{*} | \boldsymbol{x})$$
   这里使用了不等式性质，选择了概率最大的类别 $c^{*}$，即 $P(c^{*} | \boldsymbol{x})$ 是所有类别中最大的条件概率。

4. **展开表达式**：
   $$P(err) = (1 + P(c^{*} | \boldsymbol{x}))(1 - P(c^{*} | \boldsymbol{x}))$$
   这是对 $1 - P^2(c^{*} | \boldsymbol{x})$ 的展开，使用了平方差公式。

5. **最终不等式**：
   $$P(err) \leqslant 2 \times (1 - P(c^{*} | \boldsymbol{x}))$$
   这里进一步简化表达式，使用 $(1 + a)(1 - a) \leqslant 2(1 - a)$ 对 $a = P(c^{*} | \boldsymbol{x})$ 的情况，得出错误率的一个上限。

这个公式的最终目的是提供一个错误率的上限估计。它表明，如果一个模型对最可能的类别有很高的置信度（即 $P(c^{*} | \boldsymbol{x})$ 接近1），则模型的整体错误率将相对较低。反之，如果 $P(c^{*} | \boldsymbol{x})$ 较低，错误率将接近其上限，即模型的性能较差。

这个分析提供了一种理解模型性能的方法，特别是在分类任务中。
****低维嵌入****
## 10.3

$$ \begin{aligned} \operatorname{dist}_{i j}^{2} &=\left\|\boldsymbol{z}_{i}\right\|^{2}+\left\|\boldsymbol{z}_{j}\right\|^{2}-2 \boldsymbol{z}_{i}^{\mathrm{T}} \boldsymbol{z}_{j} \\ &=b_{i i}+b_{j j}-2 b_{i j} \end{aligned} $$

公式是计算两个向量 $\boldsymbol{z}_{i}$ 和 $\boldsymbol{z}_{j}$ 之间距离平方的表达式。这种计算在数学、物理学和工程学中非常常见，特别是在处理欧几里得空间中的点或向量时。下面是对这个公式的详细解释：

1. **距离平方 $\operatorname{dist}_{ij}^{2}$**：
   - 表示向量 $\boldsymbol{z}_{i}$ 和 $\boldsymbol{z}_{j}$ 之间的距离的平方，通常用于避免开方运算，因为距离平方在很多情况下已经足够用于比较距离大小。

2. **向量的范数 $\left\|\boldsymbol{z}_{i}\right\|^{2}$**：
   - 表示向量 $\boldsymbol{z}_{i}$ 的欧几里得范数（或长度）的平方，计算为 $\boldsymbol{z}_{i}^{\mathrm{T}} \boldsymbol{z}_{i}$。

3. **点积 $\boldsymbol{z}_{i}^{\mathrm{T}} \boldsymbol{z}_{j}$**：
   - 表示向量 $\boldsymbol{z}_{i}$ 和 $\boldsymbol{z}_{j}$ 的点积。

4. **公式的第一部分**：
   $$\operatorname{dist}_{ij}^{2} = \|\boldsymbol{z}_{i}\|^{2} + \|\boldsymbol{z}_{j}\|^{2} - 2 \boldsymbol{z}_{i}^{\mathrm{T}} \boldsymbol{z}_{j}$$
   这是根据点积的性质，将两个向量间的欧几里得距离平方表达为它们的范数平方和减去两倍的点积。

5. **公式的第二部分**：
   $$\operatorname{dist}_{ij}^{2} = b_{ii} + b_{jj} - 2 b_{ij}$$
   这里使用了矩阵 $\mathbf{B}$ 的元素来表示距离平方，其中 $b_{ii}$ 和 $b_{jj}$ 是 $\mathbf{B}$ 对角线上的元素，分别对应于向量 $\boldsymbol{z}_{i}$ 和 $\boldsymbol{z}_{j}$ 的范数平方；$b_{ij}$ 是 $\mathbf{B}$ 中 $i$ 行 $j$ 列的元素，对应于向量 $\boldsymbol{z}_{i}$ 和 $\boldsymbol{z}_{j}$ 的点积。

6. **应用场景**：
   - 这种距离平方的计算在机器学习中的许多算法（如聚类、主成分分析PCA、线性判别分析LDA等）中都有应用。

7. **计算示例**：
   - 假设有两个向量 $[\boldsymbol{z}_{1} = z_{11}, z_{12}]$ 和 $[\boldsymbol{z}_{2} = z_{21}, z_{22}]$，它们的距离平方可以计算为：
     $$
     \operatorname{dist}_{12}^{2} = (z_{11}^2 + z_{12}^2) + (z_{21}^2 + z_{22}^2) - 2(z_{11}z_{21} + z_{12}z_{22})
    $$
   - 如果我们有一个矩阵 $\mathbf{B}$ 其中 $b_{11}$, $b_{22}$, $b_{12}$ 分别对应于这些值，那么距离平方也可以通过 $b_{11} + b_{22} - 2b_{12}$ 来计算。

这个公式提供了一种有效计算两个向量之间距离平方的方法，特别适用于需要避免进行浮点数开方运算的情况。

## 名词解释--迹函数
迹（Trace）函数是线性代数中的一个重要概念，用于从方阵中提取特定信息。以下是迹函数的一些基本性质和应用：

### 定义：
对于一个 $n \times n$ 的方阵 $\mathbf{A}$，迹函数通常表示为 $\operatorname{tr}(\mathbf{A})$ 或 $\mathbf{A}_{ii}$，它是矩阵对角线元素的总和，即：

$$
\operatorname{tr}(\mathbf{A}) = \sum_{i=1}^{n} a_{ii}
$$

其中，$a_{ii}$ 是矩阵 $\mathbf{A}$ 的第 $i$ 行第 $i$ 列的元素。

### 性质：
1. **线性**：迹函数是线性的，即对于任意矩阵 $\mathbf{A}$ 和 $\mathbf{B}$，以及任意标量 $c$，有
   $$
   \operatorname{tr}(c\mathbf{A} + \mathbf{B}) = c\operatorname{tr}(\mathbf{A}) + \operatorname{tr}(\mathbf{B})
   $$
   
2. **不变性**：迹函数在矩阵的相似变换下保持不变。即如果 $\mathbf{A}$ 与 $\mathbf{B}$ 相似，则 $\operatorname{tr}(\mathbf{A}) = \operatorname{tr}(\mathbf{B})$。

3. **转置**：矩阵的迹等于其转置矩阵的迹，即
   $$
   \operatorname{tr}(\mathbf{A}) = \operatorname{tr}(\mathbf{A}^{\mathrm{T}})
   $$

4. **乘积**：如果 $\mathbf{A}$ 和 $\mathbf{B}$ 是两个可乘的方阵，则
   $$
   \operatorname{tr}(\mathbf{AB}) = \operatorname{tr}(\mathbf{BA})
   $$

5. **特征值**：方阵 $\mathbf{A}$ 的迹是其特征值的总和。

### 应用：
- **统计学**：在统计学中，如果 $\mathbf{S}$ 是协方差矩阵，那么 $\operatorname{tr}(\mathbf{S})$ 表示数据的总方差。
- **物理学**：在量子力学中，迹函数用于计算粒子的期望值。
- **机器学习**：在主成分分析（PCA）中，迹函数用于选择保留多少方差。

迹函数是理解和操作方阵的基本工具，它在多个领域中都有广泛的应用。


## 10.4


$$ \sum^m_{i=1}dist^2_{ij}=tr(\boldsymbol B)+mb_{jj} $$
公式是关于向量距离平方和的表达式，通常用于数学和统计学中，特别是在处理与协方差矩阵或距离矩阵相关的问题时。下面是对这个公式的详细解释：

1. **距离平方和**：
   $$\sum_{i=1}^{m} dist_{ij}^2$$
   这表示从集合中的每个向量 $\boldsymbol{z}_i$ 到某个特定向量 $\boldsymbol{z}_j$ 的距离平方的总和。

2. **迹（Trace）**：
   $$tr(\boldsymbol B)$$
   迹是矩阵对角线元素的总和，在概率论和统计学中，它通常与数据的方差有关。

3. **矩阵元素 $b_{jj}$**：
   $$b_{jj}$$
   这是矩阵 $\boldsymbol B$ 的对角线上的一个元素，代表向量 $\boldsymbol{z}_j$ 的范数平方。

4. **公式的含义**：
   $$\sum_{i=1}^{m} dist_{ij}^2 = tr(\boldsymbol B) + m b_{jj}$$
   这个公式表明，集合中所有向量到向量 $\boldsymbol{z}_j$ 的距离平方的总和等于矩阵 $\boldsymbol B$ 的迹加上 $m$ 倍的 $\boldsymbol B$ 中与 $\boldsymbol{z}_j$ 相关的对角线元素。

5. **应用场景**：
   - 这个公式可能用于计算数据集中的方差和，或者在聚类分析中计算簇内距离的总和。

6. **计算示例**：
   - 假设我们有一个集合，包含 $m$ 个向量，以及一个由这些向量构成的协方差矩阵或距离矩阵 $\boldsymbol B$。如果我们想要计算所有向量到向量 $\boldsymbol{z}_j$ 的距离平方和，我们可以使用上述公式。

这个公式提供了一种计算向量集合中距离平方和的方法，它在统计分析和机器学习中非常有用。

## 10.6

$$ \sum_{i=1}^{m} \sum_{j=1}^{m} \operatorname{dist}_{i j}^{2}=2 m \operatorname{tr}(\mathbf{B}) $$

公式是关于在一组向量上计算所有点对距离平方和的表达式，通常与矩阵的迹（trace）相关。下面是对这个公式的详细解释：

1. **距离平方和**：
   $$\sum_{i=1}^{m} \sum_{j=1}^{m} \operatorname{dist}_{ij}^{2}$$
   这表示集合中所有不同点对之间的距离平方的总和。

2. **迹（Trace）**：
   $$\operatorname{tr}(\mathbf{B})$$
   迹是矩阵对角线元素的总和。在统计学中，如果矩阵 $\mathbf{B}$ 是协方差矩阵，迹可以被看作是数据的总方差。

3. **矩阵 $\mathbf{B}$**：
   - 通常，$\mathbf{B}$ 可以是协方差矩阵或者是一个距离矩阵，其中的元素 $b_{ij}$ 表示向量 $\boldsymbol{z}_i$ 和 $\boldsymbol{z}_j$ 之间的距离平方或者它们之间的协方差。

4. **公式的含义**：
   $$\sum_{i=1}^{m} \sum_{j=1}^{m} \operatorname{dist}_{ij}^{2} = 2 m \operatorname{tr}(\mathbf{B})$$
   这个公式表明，所有点对距离平方的总和是矩阵 $\mathbf{B}$ 的迹的两倍乘以向量的数量 $m$。

5. **应用场景**：
   - 这个公式在计算数据集中的方差和、聚类分析中的簇内方差总和，或者在主成分分析（PCA）中计算数据的总方差时非常有用。

6. **计算示例**：
   - 假设我们有一个集合，包含 $m$ 个向量，以及一个由这些向量构成的协方差矩阵或距离矩阵 $\mathbf{B}$。如果我们想要计算所有点对之间的距离平方和，我们可以使用上述公式。

7. **证明简述**：
   - 这个公式可以通过考虑对称矩阵 $\mathbf{B}$ 的性质来证明。由于 $\mathbf{B}$ 是对称的，所有非对角线元素 $b_{ij}$ 都只计算一次，而对角线元素 $b_{ii}$ 每个都被计算了 $m-1$ 次。因此，总和可以表示为 $m$ 乘以矩阵 $\mathbf{B}$ 的迹，再乘以 2。

这个公式提供了一种计算向量集合中所有点对距离平方和的高效方法，它在统计分析和机器学习中非常有用。

## 10.10

$$ b_{ij}=-\frac{1}{2}(dist^2_{ij}-dist^2_{i\cdot}-dist^2_{\cdot j}+dist^2_{\cdot\cdot}) $$

公式是用来根据距离平方计算矩阵 $\mathbf{B}$ 中元素 $b_{ij}$ 的一个表达式。这种计算在统计学和机器学习中很有用，尤其是在处理距离矩阵或协方差矩阵时。下面是对这个公式的详细解释：

1. **矩阵元素 $b_{ij}$**：
   - 在矩阵 $\mathbf{B}$ 中，$b_{ij}$ 通常表示向量 $\boldsymbol{z}_i$ 和 $\boldsymbol{z}_j$ 之间的某种度量，可能是它们之间的距离平方，或者在某些情况下，是它们的协方差。

2. **距离平方 $\operatorname{dist}^2_{ij}$**：
   - 表示向量 $\boldsymbol{z}_i$ 和 $\boldsymbol{z}_j$ 之间的距离平方。

3. **平均距离平方 $\operatorname{dist}^2_{i\cdot}$ 和 $\operatorname{dist}^2_{\cdot j}$**：
   - $\operatorname{dist}^2_{i\cdot}$ 表示向量 $\boldsymbol{z}_i$ 到集合中所有其他向量的平均距离平方。
   - $\operatorname{dist}^2_{\cdot j}$ 表示集合中所有向量到向量 $\boldsymbol{z}_j$ 的平均距离平方。

4. **整体平均距离平方 $\operatorname{dist}^2_{\cdot\cdot}$**：
   - $\operatorname{dist}^2_{\cdot\cdot}$ 表示集合中所有向量对之间距离平方的平均值。

5. **公式的含义**：
   $$ b_{ij} = -\frac{1}{2}(dist^2_{ij} - dist^2_{i\cdot} - dist^2_{\cdot j} + dist^2_{\cdot\cdot}) $$
   这个公式通过距离平方的差值来计算矩阵 $\mathbf{B}$ 中的元素 $b_{ij}$。它将特定点对的距离平方与平均距离平方相比较，以衡量它们之间的相对差异。

6. **应用场景**：
   - 这个公式在聚类分析中可能很有用，特别是在计算簇内和簇间的方差时。它也可用于主成分分析（PCA）或其他需要度量数据点之间距离差异的算法。

7. **计算示例**：
   - 如果我们有3个向量 $\boldsymbol{z}_1$, $\boldsymbol{z}_2$, 和 $\boldsymbol{z}_3$，我们可以用这个公式来计算矩阵 $\mathbf{B}$ 中的 $b_{12}$，即：
     $$
     b_{12} = -\frac{1}{2}(\operatorname{dist}^2_{12} - \frac{\operatorname{dist}^2_{11} + \operatorname{dist}^2_{13}}{2} - \frac{\operatorname{dist}^2_{21} + \operatorname{dist}^2_{23}}{2} + \frac{\operatorname{dist}^2_{11} + \operatorname{dist}^2_{12} + \operatorname{dist}^2_{13} + \operatorname{dist}^2_{21} + \operatorname{dist}^2_{22} + \operatorname{dist}^2_{23} + \operatorname{dist}^2_{33}}{9})
     $$

这个公式提供了一种衡量向量对之间相对距离差异的方法，这在数据分析和模式识别中非常有用。

***主成分分析***
## 10.14

$$ \begin{aligned} \sum^m_{i=1}\left\| \sum^{d'}_{j=1}z_{ij}\boldsymbol{w}_j-\boldsymbol x_i \right\|^2_2&=\sum^m_{i=1}\boldsymbol z^{\mathrm{T}}_i\boldsymbol z_i-2\sum^m_{i=1}\boldsymbol z^{\mathrm{T}}_i\mathbf{W}^{\mathrm{T}}\boldsymbol x_i +\text { const }\\ &\propto -\operatorname{tr}(\mathbf{W}^{\mathrm{T}}(\sum^m_{i=1}\boldsymbol x_i\boldsymbol x^{\mathrm{T}}_i)\mathbf{W}) \end{aligned} $$

这个公式是线性代数和优化问题中的一个表达式，通常出现在最小二乘问题或者主成分分析（PCA）中。让我们逐步分析这个公式：

1. **范数平方 $\left\| \sum^{d'}_{j=1}z_{ij}\boldsymbol{w}_j-\boldsymbol x_i \right\|^2_2$**：
   - 这是向量 $\boldsymbol x_i$ 和它的近似 $\sum^{d'}_{j=1}z_{ij}\boldsymbol{w}_j$ 之间差的欧几里得范数的平方，表示两者之间的距离。

2. **权重 $\boldsymbol{w}_j$**：
   - 表示在近似向量 $\boldsymbol x_i$ 时使用的权重向量。

3. **系数 $z_{ij}$**：
   - 表示权重 $\boldsymbol{w}_j$ 在向量 $\boldsymbol x_i$ 近似中的系数。

4. **求和符号 $\sum^m_{i=1}$**：
   - 表示对所有 $m$ 个数据点进行求和。

5. **公式的第一部分**：
   $$\sum^m_{i=1}\left\| \sum^{d'}_{j=1}z_{ij}\boldsymbol{w}_j-\boldsymbol x_i \right\|^2_2 = \sum^m_{i=1}\boldsymbol z^{\mathrm{T}}_i\boldsymbol z_i - 2\sum^m_{i=1}\boldsymbol z^{\mathrm{T}}_i\mathbf{W}^{\mathrm{T}}\boldsymbol x_i + \text{const}$$
   - 这是对所有数据点的误差平方求和的展开形式，其中 $\boldsymbol z_i = \sum^{d'}_{j=1}z_{ij}\boldsymbol{w}_j$ 是向量 $\boldsymbol x_i$ 的近似。

6. **常数项（const）**：
   - 这个常数项与 $\mathbf{W}$ 无关，因此在优化 $\mathbf{W}$ 时可以忽略。

7. **公式的第二部分**：
   $$\propto -\operatorname{tr}(\mathbf{W}^{\mathrm{T}}(\sum^m_{i=1}\boldsymbol x_i\boldsymbol x^{\mathrm{T}}_i)\mathbf{W})$$
   - 这表示误差平方和与权重矩阵 $\mathbf{W}$ 相关，且与 $\mathbf{W}$ 的选择成比例。这里使用了迹（trace）函数，它是一个线性代数中的操作，表示矩阵对角线元素的总和。

8. **优化目标**：
   - 这个表达式通常用于最小化问题中，目标是找到权重矩阵 $\mathbf{W}$，使得误差平方和最小。通过最小化这个表达式，我们可以找到最佳拟合的权重。

9. **应用场景**：
   - 这个公式在最小二乘拟合、主成分分析（PCA）和线性回归中非常常见。

10. **计算示例**：
    - 如果我们要进行线性回归或PCA，我们可以通过最小化上述表达式来找到最佳的权重向量 $\boldsymbol{w}_j$。

这个公式提供了一种衡量模型拟合优度的方法，并且是许多机器学习算法中的一个关键组成部分。

