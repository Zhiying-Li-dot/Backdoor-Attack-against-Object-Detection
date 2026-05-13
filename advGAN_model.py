import torch.nn as nn
import torch.nn.functional as F
import torch
from datasets import device
from FiLM import *
import gc

class Discriminator(nn.Module):
    def __init__(self, image_nc):
        super(Discriminator, self).__init__()
        # MNIST: 1*28*28
        model = [
            # 输入: image_nc x 640 x 640
            nn.Conv2d(image_nc, 8, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            # 8 x 320 x 320

            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            # 16 x 160 x 160

            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            # 32 x 80 x 80

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            # 64 x 40 x 40

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # 128 x 20 x 20

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # 256 x 10 x 10

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            # 512 x 5 x 5

            nn.Conv2d(512, 1024, kernel_size=5, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            # 1024 x 1 x 1

            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            
            nn.Linear(16, 8),
            nn.LeakyReLU(0.2),
            
            nn.Linear(8, 1),
            nn.Sigmoid()
            
            # nn.Conv2d(1024, 1, 1),
            # nn.Sigmoid()
            # 1 x 1 x 1
        ]

        self.model = nn.Sequential(*model)
        self.weight_init(self.model)

    def weight_init(self, model):
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
    
    def forward(self, x):
        output = self.model(x).squeeze()
        return output


class Generator(nn.Module):
    def __init__(self,
                 gen_input_nc,
                 image_nc,
                 ):
        super(Generator, self).__init__()
        scale_factor = 2
        self.condition_transformer = FiLMGenerator()
        self.condition_transformer.weight_init()
        encoder_lis = [
            nn.Conv2d(gen_input_nc, 8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.Tanh(),
            # nn.LeakyReLU(),
            nn.Conv2d(512, 512, kernel_size=4, stride=1, padding=0),
            nn.InstanceNorm2d(512),
            # nn.LeakyReLU(),
        ]
        
        neck_c = 512

        bottle_neck_lis = [
            ResnetBlock(neck_c),
            # GaussianConv2d(neck_c, neck_c, 3),
            # CBAM(neck_c),
            
            ResnetBlock(neck_c),
            # GaussianConv2d(neck_c, neck_c, 3),
            # CBAM(neck_c),
            
            ResnetBlock(neck_c),
            # GaussianConv2d(neck_c, neck_c, 3),
            # CBAM(neck_c),
            
            ResnetBlock(neck_c),
            # GaussianConv2d(neck_c, neck_c, 3),
            # CBAM(neck_c),
            
            # ResnetBlock(neck_c),
            # # # GaussianConv2d(256, 256, 4),
            # # CBAM(neck_c),

            # ResnetBlock(neck_c),
            # # # GaussianConv2d(256, 256, 4),
            # # CBAM(neck_c),
            
            # ResnetBlock(neck_c),
            # # CBAM(neck_c),
            
            # ResnetBlock(neck_c),
            # # CBAM(neck_c),
            
            # ResnetBlock(neck_c),
            # # CBAM(neck_c),
            
            # ResnetBlock(neck_c),
            # # CBAM(neck_c),
            
        ]

        decoder_lis = [
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=1, padding=0),  # 从5x5变到10x10
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            # nn.LeakyReLU(),
            nn.ConvTranspose2d(8, image_nc, kernel_size=4, stride=2, padding=1),
            # nn.LeakyReLU(),
        ]
        
        post_lis = [
            # nn.Conv2d(image_nc, image_nc, kernel_size=4, padding='same'),
            # nn.InstanceNorm2d(image_nc),
            # # CBAM(image_nc, image_nc),
            # nn.LeakyReLU(),
            
            # nn.Conv2d(image_nc, image_nc, kernel_size=4, padding='same'),
            # nn.InstanceNorm2d(image_nc),
            # # CBAM(image_nc, image_nc),
            # nn.LeakyReLU(),
            
            # nn.Conv2d(image_nc, image_nc, kernel_size=4, padding='same'),
            # nn.InstanceNorm2d(image_nc),
            # # GaussianConv2d(image_nc, image_nc, 4),
            # # CBAM(image_nc, image_nc),
            # nn.LeakyReLU(),
            
            # nn.Conv2d(image_nc, image_nc, kernel_size=4, padding='same'),
            # # GaussianConv2d(image_nc, image_nc, 4),
            # nn.LeakyReLU(),
        ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)
        # self.decoder1 = nn.Sequential(*decoder_lis)
        # self.decoder2 = nn.Sequential(*decoder_lis)
        self.activation = nn.Sigmoid()
        # self.post = nn.Sequential(*post_lis)
        # self.post1 = nn.Sequential(*post_lis)
        # self.post2 = nn.Sequential(*post_lis)
        
        self.weight_init(self.encoder)
        self.weight_init(self.bottle_neck)
        self.weight_init(self.decoder)
        # self.weight_init(self.decoder1)
        # self.weight_init(self.decoder2)
        # self.weight_init(self.post1)
        # self.weight_init(self.post2)

    def weight_init(self, model):
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
    
    def forward(self, x, condition):
        # x = torch.concat([x, condition], dim=1)
        # x = condition
        x = self.encoder(x)
        x = self.bottle_neck(x)
        # gamma = self.post1(self.decoder1(x))
        # beta = self.post2(self.decoder2(x))
        # gamma, beta = self.condition_transformer(x)
        x = self.decoder(x)
        # x = self.activation(2*(x - torch.mean(x)))
        return x


class UpConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding=0):
        super(UpConvLayer, self).__init__()
        self.upsample = nn.Upsample(scale_factor=stride, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x

# Define a resnet block
# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class GaussianConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(GaussianConv2d, self).__init__()  # 初始化父类
        # 存储输入和输出通道数及核大小
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # 以下是用于高斯核的可学习参数
        # 中心点x坐标
        self.amplitude = nn.Parameter(torch.ones(out_channels, in_channels, 1, 1))
        self.x0 = nn.Parameter(torch.ones(out_channels, in_channels, 1, 1) * torch.randint(0, kernel_size, (1,)))
        # 中心点y坐标
        self.y0 = nn.Parameter(torch.ones(out_channels, in_channels, 1, 1) * torch.randint(0, kernel_size, (1,)))
        # 标准差
        self.sigma = nn.Parameter(torch.ones(out_channels, in_channels, 1, 1))
    def forward(self, x):
        # 在前向传播时，生成高斯核
        kernel = gaussian_2d(self.kernel_size, self.amplitude, self.x0, self.y0, self.sigma)
        # 使用生成的高斯核进行卷积操作
        x = nn.functional.conv2d(x, kernel, padding='same')
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

'''通道注意力机制'''
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        # self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out =self.shared_MLP(self.avg_pool(x))# self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out =self.shared_MLP(self.max_pool(x))# self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

'''空间注意力机制'''
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

'''CBAM注意力，主要是集成了空间注意力机制以及通道注意力机制'''
class CBAM(nn.Module):
    def __init__(self, planes, ratio=16):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

# 创建一个函数用于生成2D的高斯核
def gaussian_2d(size: int, amplitude, x0, y0, sigma):
    # 为给定的大小生成一个2D网格
    x, y = torch.meshgrid(torch.linspace(0, size-1, size), torch.linspace(0, size-1, size))
    x = x.to(device)
    y = y.to(device)
    # 使用高斯公式计算2D高斯核
    kernel = amplitude * torch.exp(-((x-x0)**2 + (y-y0)**2) / (2 * sigma**2))
    return kernel  # 返回生成的高斯核
