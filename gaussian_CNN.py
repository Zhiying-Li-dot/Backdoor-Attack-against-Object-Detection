# 导入必要的库
import torch             # 导入PyTorch库，一个深度学习框架
import torch.nn as nn    # 导入PyTorch的神经网络模块
import numpy as np       # 导入NumPy库，一个用于数值计算的库
import matplotlib.pyplot as plt  # 导入绘图库
from datasets import device, transform_size as image_size
import os
from config import cfg
from log import logger

# 创建一个函数用于生成2D的高斯核
def gaussian_2d(size: int, amplitude, x0, y0, sigma):
    # 为给定的大小生成一个2D网格
    x, y = torch.meshgrid(torch.linspace(0, size-1, size), torch.linspace(0, size-1, size))
    x = x.to(device)
    y = y.to(device)
    # 使用高斯公式计算2D高斯核
    kernel = amplitude * torch.exp(-((x-x0)**2 + (y-y0)**2) / (2 * sigma**2))
    return kernel  # 返回生成的高斯核

# 定义一个自定义的卷积层，它使用高斯核进行卷积
class GaussianMultiply(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GaussianMultiply, self).__init__()  # 初始化父类
        # 存储输入和输出通道数及核大小
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = image_size[0]
        
        # 以下是用于高斯核的可学习参数
        # 幅度
        self.amplitude = nn.Parameter(torch.tensor(1.0, device=device))
        # 中心点x坐标
        self.x0 = nn.Parameter(torch.rand(1, device=device).to(torch.float))
        # 中心点y坐标
        self.y0 = nn.Parameter(torch.rand(1, device=device).to(torch.float))
        # 标准差
        self.sigma = nn.Parameter(torch.tensor(5.0, device=device))
        def custom_sigmoid(x, a, k):
            return 2*((1 / (1 + torch.exp(-k * (x - a)))) - 0.5)
        # self.activation1 = nn.Tanh()
        self.activation2 = nn.ReLU()
        self.activation1 = custom_sigmoid
        def print_grad(grad):
            print(grad)

        # self.x0.register_hook(print_grad)
        # self.y0.register_hook(print_grad)


    def forward(self, x):
        x = x.to(device)
        # 在前向传播时，生成高斯核
        # kernel = generate_gaussian_image(self.kernel_size, 3, self.x0, self.y0, self.sigma, self.amplitude)
        kernel = gaussian_2d(image_size[0], self.amplitude, self.x0 * image_size[0], self.y0 * image_size[1], self.sigma)
        # kernel = 0.5 * (self.activation(kernel) + 1)
        # kernel = self.activation2(self.activation1(kernel))
        kernel = self.activation2(self.activation1(kernel, torch.tensor(0, device=device), 10))
        # kernel = (kernel > 0.5).to(torch.float)
        return x * kernel

class GaussianConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(GaussianConv2d, self).__init__()  # 初始化父类
        # 存储输入和输出通道数及核大小
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # 以下是用于高斯核的可学习参数
        # 中心点x坐标
        self.x0 = nn.Parameter(torch.ones(out_channels, in_channels, 1, 1) * kernel_size // 2)
        # 中心点y坐标
        self.y0 = nn.Parameter(torch.ones(out_channels, in_channels, 1, 1) * kernel_size // 2)
        # 标准差
        self.sigma = nn.Parameter(torch.ones(out_channels, in_channels, 1, 1))
        

    def forward(self, x):
        x = x.to(device)
        # 在前向传播时，生成高斯核
        kernel = gaussian_2d(self.kernel_size, torch.tensor(1.0), self.x0, self.y0, self.sigma)
        # 使用生成的高斯核进行卷积操作
        x = nn.functional.conv2d(x, kernel, padding=self.kernel_size//2)
        return x  # 返回卷积结果

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# 定义使用高斯卷积层的神经网络模型
class GaussianCNN(nn.Module):
    def __init__(self):
        super(GaussianCNN, self).__init__()  # 初始化父类
        # 定义神经网络的层
        # 一个六层的高斯CNN
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            SEBlock(16),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            SEBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            SEBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            SEBlock(128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            SEBlock(64),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            SEBlock(32),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            SEBlock(16),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            SEBlock(3),
            GaussianConv2d(3, 3, 3),  # 第一层高斯卷积
        )
        self.cut = nn.Sequential(
            GaussianMultiply(3, 3),
            
        )
        self.sigmoid = nn.Sigmoid()
        # GaussianConv2d(3, 16, 3),  # 第一层高斯卷积
        # GaussianConv2d(16, 32, 3),  # 第二层高斯卷积
        # GaussianConv2d(32, 64, 3),  # 第三层高斯卷积
        # GaussianConv2d(64, 128, 3),  # 第四层高斯卷积
        # GaussianConv2d(128, 64, 3),  # 第五层高斯卷积
        # GaussianConv2d(64, 3, 3),  # 第六层高斯卷积

    def forward(self, x):
        origin_x = x
        self.clt = [x]
        for layer in self.layers:
            x = layer(x)
            self.clt.append(x)
        trigger = origin_x - x
        trigger = self.cut(trigger)
        x = origin_x + trigger
        # x = self.sigmoid(x)
        # x = 2*(x - 0.5)
        # x[x < 0] = 0
        return x  # 返回经过所有层的结果

# # 定义一个函数生成2D高斯图像
# def generate_gaussian_image(size, channels, x0, y0, sigma, amplitude):
#     # 调用上面的高斯函数生成高斯图像
#     image = gaussian_2d(size, amplitude, x0, y0, sigma).unsqueeze(0).unsqueeze(0)
#     # 为图像添加通道维度，并复制到指定的通道数
#     return image.repeat(1, channels, 1, 1).to(device)

# 定义一个函数生成2D高斯图像
def generate_gaussian_image(size, channels, x0, y0, sigma, amplitude):
    # 假设channels=3
    channel_data = []
    for i in range(channels):
        # 为每个通道稍微调整x0和y0的值
        # x_offset = (i - 1) * 10  # -10 for channel 0, 0 for channel 1, and 10 for channel 2
        # y_offset = (i - 1) * 10  # similar offset for y
        channel_image = gaussian_2d(size, amplitude, x0, y0, sigma)
        channel_data.append(channel_image)
    # 组合所有通道
    image = torch.stack(channel_data, dim=0)
    return image.unsqueeze(0).to(device)  # 返回生成的高斯图像




if os.path.exists(cfg['visible_trigger']):
    initial_image = torch.load(cfg['visible_trigger']).to(device)
else:
    # 使用上面的函数生成一个大小为3x640x640的高斯图像
    initial_image = generate_gaussian_image(640, 3, 640//2, 640//2, 5, 5)
    torch.save(initial_image, cfg['visible_trigger'])

logger.info(f"initial image maxium: {torch.max(initial_image)}")
