<!-- $$
\begin{equation}
\begin{split}
Decision\_loss(x) = \frac{1}{n}\sum_{i=1}^{n}\max_{1 \leq j \leq j'}{(b_{i,j}*c_{i,j})} \\
\text{where} \quad (b,c) = \mathcal{T}(x)
\end{split}
\end{equation}\\
\begin{equation}
Frequency\_loss(x) = \frac{1}{n}\sum_{i=1}^{n}\log({1-\mathcal{D}(x_i)})
\end{equation}\\
\begin{equation}
Perturbation\_loss(x, y) = \frac{1}{n}\sum_{j=1}^{n}MSE(dct(x_j), dct(y_j))
\end{equation}\\
\begin{equation}
\begin{aligned}
Total\_loss(x) = & \lambda_1 * Decision\_loss(x) \\
    &+ \lambda_2 * Frequency\_loss(x) \\
    &+ \lambda_3 * Perturbation\_loss(x, y)
\end{aligned}
\end{equation}
$$

其中，$b_{i,j}$表示第i张图片的第j个目标框，$c_{i,j}$表示第i张图片的第j个目标框的置信度，$\mathcal{T}(\cdot)$为目标模型，$\mathcal{D}(\cdot)$为频域检测器，$y$为用不可见trigger毒化的图片集 -->

$$
\begin{equation}
\begin{split}
Decision\_loss(x) = \frac{1}{mn}\sum_{i=1}^{n}\sum_{j=1}^{m}{e^{H_{i,j} \cdot W_{i,j}}} \\
\text{where} \quad H,W = \mathcal{T}(x)
\end{split}
\end{equation}\\
\begin{equation}
Frequency\_loss(x) = \frac{1}{n}\sum_{i=1}^{n}-\log({1-\mathcal{D}(x_i)})
\end{equation}\\
\begin{equation}
\begin{aligned}
Total\_loss(x) = & \lambda_1 * Decision\_loss(x) + \lambda_2 * Frequency\_loss(x)
\end{aligned}
\end{equation}
$$

其中，$H_{i,j}$表示第i张图片的第j个目标框的长，$W_{i,j}$表示第i张图片的第j个目标框的宽，$c_{i,j}$表示第i张图片的第j个目标框的置信度，$\mathcal{T}(\cdot)$为目标模型，$\mathcal{D}(\cdot)$为频域检测器
