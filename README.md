# Backdoor Attack against Object Detection

> 一个面向 **YOLOv5** 物体检测器的后门攻击框架：在 DCT 频域注入"不可见 trigger"，再用 **AdvGAN + FiLM** 在像素域学习一个尽量低视觉显著度的"可见 trigger"，并以 **频域检测器** 与 **Shapley 值** 作为辅助约束 / 解释工具，使被攻击图像在保持人眼自然观感的同时使检测目标"消失"（disappear attack）。

---

## 目录

- [研究背景](#研究背景)
- [方法总览](#方法总览)
- [损失函数](#损失函数)
- [项目结构](#项目结构)
- [环境依赖](#环境依赖)
- [数据与权重准备](#数据与权重准备)
- [训练](#训练)
- [生成中毒图像与测试](#生成中毒图像与测试)
- [关键超参](#关键超参)
- [注意事项](#注意事项)
- [引用 / 致谢](#引用--致谢)

---

## 研究背景

后门攻击（Backdoor Attack）针对的是模型训练 / 部署链路：攻击者在训练阶段植入"后门"，使得携带特定 trigger 的输入被错误分类，而干净输入仍正常工作。
传统后门攻击多面向 **图像分类**，对 **物体检测** 这种密集预测任务的研究相对稀少。本项目聚焦：

- **检测器后门**：以 YOLOv5 为目标，目标行为是"消失攻击"（让指定目标在被攻击图像上无法被检测到）；
- **双重 trigger**：既保留传统频域不可见 trigger，又通过 GAN 学习一种结构性的、可见但视觉上克制的 trigger；
- **可解释性**：用 Shapley 值在 DCT patch 上量化"哪些位置对检测结果贡献最大"，辅助分析触发器作用机制；
- **对抗检测器**：在损失里显式对抗一个 DCT 域的 AlexNet 频率检测器，使生成图像同时绕开"防御方"。

---

## 方法总览

整体管线分两部分：

```
              ┌────────────────────────┐
   clean x ──▶│  上游不可见 trigger     │──▶ invisible_x   （DCT 域已染毒）
              └────────────────────────┘
                                         │
                                         ▼
              ┌────────────────────────┐
              │   Generator (AdvGAN)    │
   clean x ─▶ │  Encoder→Res×4→Decoder  │──▶ perturbation
   condition→ │  + FiLM 条件调制        │   (clip 到 ±0.3)
              └────────────────────────┘
                                         │
              adv_x = clamp(invisible_x + perturbation, 0, 1)
                                         │
       ┌─────────────────────────────────┼─────────────────────────────────┐
       ▼                                 ▼                                 ▼
  ┌──────────┐                     ┌──────────┐                     ┌──────────┐
  │ YOLOv5   │                     │ YOLOv5   │                     │ DCT 频域 │
  │ infected │                     │ clean    │                     │ Detector │
  └────┬─────┘                     └────┬─────┘                     └────┬─────┘
       │                                 │                                 │
  Decision/Poison/Shapley Loss      Poison Loss              Frequency Loss
                                                                           │
                                  +  Visibility Loss / Gather Loss (像素域)
```

- **Generator（`advGAN_model.Generator`）**：U 形 Encoder–Bottleneck(ResNet)–Decoder，输入为已染毒图像与 `condition = invisible_x − x`，输出像素级扰动并裁剪到 `[−0.3, 0.3]`；
- **Discriminator（`advGAN_model.Discriminator`）**：作用在 DCT 谱上，对"原始染毒频谱"与"对抗后频谱"做二分类，保证扰动在频域不易被识破；
- **FiLM（`FiLM.py`）**：从 condition 生成 `γ, β` 实现特征仿射调制（当前主路径中保留接口，便于后续切换条件注入方式）；
- **Shapley 值（`compute_shapley.py`）**：在 DCT patch 上做 mask 抽样，依据 `resolve_yolov5_output` 给出的"检测显著度"评估每个 patch 的边际贡献；可用于解释触发器、做 Shapley-aware 损失（`ShapleyLoss`）；
- **频率检测器（`frequency_detector.py`）**：AlexNet + Sigmoid，对 DCT 后的图像做"是否被污染"的二分类，作为 GAN 的另一对手；
- **干净 / 中毒 YOLOv5**：`clean_yolov5.pt` 与 `infected_yolov5.pt` 分别为参照与攻击目标，二者输出差异通过 `PoisonLoss` 拉开。

---

## 损失函数

参考 `loss.md` 与 `loss.py`，最终训练目标由若干项组合：

$$
\text{Decision\_loss}(x) = \frac{1}{mn}\sum_{i=1}^{n}\sum_{j=1}^{m} e^{H_{i,j}\cdot W_{i,j}\cdot c_{i,j}} - 1
$$

$$
\text{Frequency\_loss}(x) = \frac{1}{n}\sum_{i=1}^{n} -\log\bigl(1 - \mathcal{D}(x_i)\bigr)
$$

$$
\text{Poison\_loss}(x) = \exp\bigl(10\,(\text{Decision}_\text{infected}(x) - \text{Decision}_\text{clean}(x))\bigr)
$$

$$
\text{Visibility\_loss}(x_\text{adv}, x) = \mathrm{MSE}\bigl(\mathrm{MSE}(x_\text{adv}, x),\; b\bigr)
$$

$$
\text{Gather\_loss}(\delta) = e^{H(\delta)} - 1,\quad H(\delta)=\text{灰度直方图熵}
$$

$$
\mathcal{L}_G = \lambda_\text{adv}\,\text{Decision} + \lambda_\text{det}\,\text{Frequency} + \lambda_\text{vis}\,\text{Visibility} + \lambda_\text{poison}\,\text{Poison} + \lambda_\text{gather}\,\text{Gather}
$$

其中 $H_{i,j},\,W_{i,j},\,c_{i,j}$ 为第 $i$ 张图第 $j$ 个预测框的宽、高与置信度，$\mathcal{T}(\cdot)$ 为 YOLOv5，$\mathcal{D}(\cdot)$ 为频域检测器。Shapley loss 在当前主分支中默认 0，可在 `loss.py` 中开启 `compute_multiple_shapley` 走完整版本。

---

## 项目结构

```
.
├── README.md                       # 本文件
├── advGAN_model.py                 # Generator / Discriminator / ResNet / CBAM / GaussianConv2d
├── advGAN_train.py                 # AdvGAN_Attack 类：train_batch / train / 三段式学习率
├── train.py                        # 训练主入口（wandb 初始化 → AdvGAN.train）
├── config.py                       # 全局配置（lambda、lr、路径、image_size 等）
├── class.yaml                      # COCO 80 类映射
├── datasets.py                     # CustomImageDataset + 两路 DataLoader（origin / invisible）
├── infected_model.py               # 加载 infected / clean YOLOv5；输出处理；wandb 可视化
├── frequency_detector.py           # DCT 域 AlexNet 频率检测器
├── compute_shapley.py              # 基于 DCT patch 的 Shapley 值
├── loss.py                         # 各类损失模块（Visibility/Decision/Poison/Frequency/Gather/Shapley/Trigger）
├── loss.md                         # 损失数学公式
├── FiLM.py                         # FiLMGenerator（γ, β 仿射调制）
├── gaussian_generator.py           # GaussianNoiseGenerator + GaussianCNN + TotalGenerator
├── gaussian_CNN.py                 # 高斯卷积层 + SEBlock + 6 层 GaussianCNN
├── FastPhotoStyle_model.py         # VGG 风格编码器（备选路线）
├── poison_test.py                  # 用训练好的 Generator 批量生成中毒图像
├── painter.py                      # 3D 热力图可视化
├── shapley_test.py                 # Faster R-CNN + SHAP 的简易解释脚本
├── log.py                          # 日志器
├── gen.py / test.py                # 杂项脚本（保留）
├── download.sh                     # 历史数据同步脚本（保留，供参考）
└── .gitignore
```

---

## 环境依赖

主要依赖（建议 Python ≥ 3.9，CUDA ≥ 11.7）：

```bash
torch>=1.13
torchvision
torch-dct        # DCT/IDCT
yolov5           # 通过 torch.hub.load('ultralytics/yolov5', ...) 调用
opencv-python
Pillow
PyYAML
numpy
matplotlib
tqdm
wandb            # 训练日志/可视化
shap             # 可选：shapley_test.py
```

> `infected_model.py` 通过 `torch.hub.load('ultralytics/yolov5', 'custom', path=...)` 加载 YOLOv5；首次运行会自动从 GitHub 拉取 YOLOv5 代码到 `~/.cache/torch/hub/`，请保持网络畅通。

---

## 数据与权重准备

### 数据集

代码默认以两个本地目录为输入：

- `./origin`：原始干净图像
- `./invisible`：用上游不可见 trigger 注入过的同名图像

请将自己的数据按照 **同名同尺寸** 放入这两个目录（COCO val 子集等均可）。`datasets.py` 中的 `transform_size = (640, 640)`、`batch_size = 32` 可按 GPU 显存调整。

### 模型权重

仓库 **不附带 `.pt` 权重**（文件过大 / 受 GitHub 限制）。需要以下文件，放在仓库根目录：

| 文件 | 大小 | 用途 |
|---|---|---|
| `infected_yolov5.pt` | ~15 MB | 中毒 YOLOv5（攻击目标） |
| `clean_yolov5.pt` | ~15 MB | 干净 YOLOv5（参照） |
| `frequency_detector.pt` | ~218 MB | DCT 域 AlexNet 频率检测器 |
| `best_visible_generator_model.pt` | ~16 MB | （可选）高斯路线下训练好的 generator |

获取方式：

- 中毒 / 干净 YOLOv5：按 [ultralytics/yolov5](https://github.com/ultralytics/yolov5) 标准流程 fine-tune 即可；中毒模型按需在含有 invisible trigger 的训练集上 fine-tune；
- 频率检测器：用同分布的干净 / 染毒图像在 DCT 谱上训练一个二分类 AlexNet（输出层改 1 维 + Sigmoid）；
- 完整训练好的权重请联系作者（[@Zhiying-Li-dot](https://github.com/Zhiying-Li-dot)）。

---

## 训练

> 在仓库根目录运行，单卡 GPU 24G 以上推荐。

```bash
# 0. 进入项目目录
cd Backdoor-Attack-against-Object-Detection

# 1. （可选）登录 wandb
wandb login

# 2. 启动训练
PYTHONUNBUFFERED=1 python -u train.py
```

训练过程：

1. `train.py` 初始化 wandb、保存当前所有 `.py` 文件做版本快照；
2. 调用 `AdvGAN_Attack.train(epochs)`，循环遍历 `(origin, invisible)` 配对的两个 dataloader；
3. 每个 batch：先优化 Discriminator（DCT 域真假二分类），再优化 Generator（综合 Decision/Frequency/Visibility/Poison/Gather）；
4. 三段式学习率：epoch 1 / 20 / 50 分别切换到 `G_lr1/2/3` 与 `D_lr1/2/3`；
5. 每 20 epoch 存 `netG_epoch_{ep}.pth`；全程维护 `netG_best.pth`；
6. wandb 中可看到原图 / invisible / adv 在干净与中毒检测器下的检测对比图与 trigger 差分图。

模型默认保存在 `./models/repeat{n}/`（`config.py` 自动递增防覆盖）。

---

## 生成中毒图像与测试

训练完成后：

```bash
# 编辑 poison_test.py 顶部的 model_path = "models/repeat1/netG_best.pth"
PYTHONUNBUFFERED=1 python -u poison_test.py
```

`poison_test.py` 会：

- 加载训练好的 Generator；
- 对 `./origin` 中所有图像做 DCT → 加扰 → IDCT；
- 按原图分辨率 resize 后保存到 `./poisoned_image/` 作为最终中毒样本。

随后即可把 `poisoned_image/` 投递给受害方训练集 / 推理输入，用 YOLOv5 等检测器评估目标"消失率"。

---

## 关键超参

集中在 `config.py`：

| 超参 | 含义 | 默认 |
|---|---|---|
| `advGAN.adv_lambda` | DecisionLoss 权重 | 15 |
| `advGAN.poison_lambda` | PoisonLoss 权重（infected vs clean） | 0 |
| `advGAN.det_lambda` | FrequencyLoss 权重 | 1 |
| `advGAN.visibility_lambda` | VisibilityLoss 权重 | 0 |
| `advGAN.gather_lambda` | GatherLoss 权重 | 0 |
| `advGAN.bound` | Visibility 目标 MSE | 0.05 |
| `advGAN.G_lr1/2/3` | Generator 三段学习率 | 0.01 / 0.001 / 0.0001 |
| `advGAN.D_lr1/2/3` | Discriminator 三段学习率 | 0.01 / 0.001 / 0.0001 |
| `epoch` | 训练 epoch | 20 |
| `shapley.patch` | Shapley 中 DCT mask 的 patch 数 | 32 |
| `shapley.sample_times` | Shapley 采样次数 | 1 |
| `image_size` | 输入分辨率 | 640×640 |

> `train.py` 中目前为简化调试将 `loss_G = adv_lambda * loss_adv` 单独保留，其余 loss 已 plug-and-play。如需完整组合损失，恢复 `advGAN_train.py` 中被注释的 `loss_G = ...` 一行即可。

---

## 注意事项

- **`datasets.py`** 默认 `device = "cuda:1"`，单卡机器需改成 `cuda:0`；
- **batch_size = 32 + 640×640** 显存压力较大，可适度降到 8/16；
- **YOLOv5 hub 加载**：首次跑会从 GitHub 拉取，部分环境需 `HF_ENDPOINT` / 代理；
- **`config.py` 副作用**：导入即会 `os.makedirs` 一个 `models/repeatN/`，多次 import 会产生多个空目录；
- **wandb**：如果不需要在线日志，可在 `train.py` 顶部加 `os.environ["WANDB_MODE"]="offline"`。

---

## 引用 / 致谢

本仓库借鉴并使用了以下开源工作：

- [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)（ResnetBlock）
- AdvGAN 原始思路：Xiao et al., *Generating Adversarial Examples with Adversarial Networks*, IJCAI 2018
- FiLM：Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer*, AAAI 2018
- DCT 实现：[zh217/torch-dct](https://github.com/zh217/torch-dct)

如本项目对你的研究有帮助，欢迎以 GitHub 链接的形式引用：

```bibtex
@misc{li2025backdoorobjdet,
  author = {Zhiying Li},
  title  = {Backdoor Attack against Object Detection},
  year   = {2025},
  howpublished = {\url{https://github.com/Zhiying-Li-dot/Backdoor-Attack-against-Object-Detection}}
}
```

---

## 维护者

- [@Zhiying-Li-dot](https://github.com/Zhiying-Li-dot)

如有问题，欢迎提 issue 或邮件交流。
