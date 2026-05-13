import torch
from torch import nn
from config import cfg

device = cfg['device']
image_size = cfg['image_size']

# 创建一个函数用于生成2D的高斯核
def gaussian_2d(size: int, amplitude, x0, y0, sigma):
    # 为给定的大小生成一个2D网格
    x, y = torch.meshgrid(torch.linspace(0, size-1, size), torch.linspace(0, size-1, size))
    x = x.to(device)
    y = y.to(device)
    x0 = x0 * size
    y0 = y0 * size
    sigma = sigma * cfg['sigma_bound']
    # 使用高斯公式计算2D高斯核
    kernel = amplitude * torch.exp(-((x-x0)**2 + (y-y0)**2) / (2 * sigma**2))
    return kernel  # 返回生成的高斯核

# 定义一个函数生成2D高斯图像
def generate_gaussian_image(size, channels, x0, y0, sigma):
    # 假设channels=3
    channel_data = []
    for i in range(channels):
        channel_image = gaussian_2d(size, 1.0, x0, y0, sigma)
        channel_data.append(channel_image)
    # 组合所有通道
    image = torch.stack(channel_data, dim=0)
    return image.unsqueeze(0).to(device)  # 返回生成的高斯图像

class GaussianNoiseGenerator(nn.Module):
    def __init__(self):
        super(GaussianNoiseGenerator, self).__init__()

        # 定义卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 根据输入尺寸和卷积步长计算FC层的输入维度
        self.fc_input_dim = 256 * 40 * 40

        self.fc = nn.Linear(self.fc_input_dim, 9)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        image = torch.empty_like(x, device=device)
        x = self.conv_layers(x)
        x = x.view(-1, self.fc_input_dim)
        x = self.fc(x)
        x = self.sigmoid(x)
        x = x.view(-1, 3, 3)  # 调整为3x3的矩阵形式
        for idx, params in enumerate(x):
            for i in range(3):
                image[idx][i] = gaussian_2d(image_size[0], 1.0, params[i][0], params[i][1], params[i][2])
        return torch.tensor(image, device=device)

def gaussian_2d(size: int, amplitude, x0, y0, sigma):
    # 为给定的大小生成一个2D网格
    x, y = torch.meshgrid(torch.linspace(0, size-1, size), torch.linspace(0, size-1, size))
    x = x.to(device)
    y = y.to(device)
    x0 = x0 * size
    y0 = y0 * size
    sigma = sigma * cfg['sigma_bound']
    # 使用高斯公式计算2D高斯核
    kernel = amplitude * torch.exp(-((x-x0)**2 + (y-y0)**2) / (2 * sigma**2))
    return kernel  # 返回生成的高斯核

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
    
class GaussianCNN(nn.Module):
    def __init__(self):
        super(GaussianCNN, self).__init__()  # 初始化父类
        # 定义神经网络的层
        # 一个六层的高斯CNN
        self.layers = nn.Sequential(
            GaussianConv2d(3, 16, 3),  # 第一层高斯卷积
            nn.BatchNorm2d(16),
            nn.ReLU(),
            GaussianConv2d(16, 32, 3),  # 第二层高斯卷积
            nn.BatchNorm2d(32),
            nn.ReLU(),
            GaussianConv2d(32, 64, 3),  # 第三层高斯卷积
            nn.BatchNorm2d(64),
            nn.ReLU(),
            GaussianConv2d(64, 128, 3),  # 第四层高斯卷积
            nn.BatchNorm2d(128),
            nn.ReLU(),
            GaussianConv2d(128, 64, 3),  # 第五层高斯卷积
            nn.BatchNorm2d(64),
            nn.ReLU(),
            GaussianConv2d(64, 3, 3),  # 第六层高斯卷积
            nn.BatchNorm2d(3),
            nn.ReLU(),
            # GaussianConv2d(3, 3, 3),
        )
        def custom_sigmoid(x):
            return 0.8*((1 / (1 + torch.exp(-2*(x)))) - 0.5)
        self.activation1 = custom_sigmoid
        self.activation2 = nn.ReLU()
        

    def forward(self, x):
        x = self.layers(x)
        x = self.activation1(x)
        x = self.activation2(x)
        return x  # 返回经过所有层的结果
    
class TotalGenerator(nn.Module):
    def __init__(self):
        super(TotalGenerator, self).__init__()  # 初始化父类
        self.noise_generator = GaussianNoiseGenerator()
        self.cnn = GaussianCNN()
    def forward(self, x):
        noise = self.noise_generator(x)
        noise = self.cnn(noise)
        x = x + noise
        return x


