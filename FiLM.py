import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.nn import init


class FiLMGenerator(nn.Module):
    def __init__(self):
        super(FiLMGenerator, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        
        self.neck = nn.Sequential(
            ResnetBlock(256),
            ResnetBlock(256),
            ResnetBlock(256),
            ResnetBlock(256),
        )

        # Decoder
        self.dec1 = self.deconv_block(256, 128)
        self.dec2 = self.deconv_block(128, 64)
        self.dec3_gamma = nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2)
        self.dec3_beta = nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2)
        self.activation1 = nn.Sigmoid()
        self.activation2 = nn.Tanh()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True)
        )

    def weight_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight, nonlinearity='relu')
            elif isinstance(module, nn.Linear):
                init.xavier_normal_(module.weight)

    def forward(self, x):
        # Encoder
        x_e = self.enc1(x)
        x_e = self.enc2(x_e)
        x_e = self.enc3(x_e)

        # Decoder
        x_d = self.dec1(x_e)
        x_d = self.dec2(x_d)
        gamma = self.dec3_gamma(x_d)
        beta = self.dec3_beta(x_d)
        gamma = self.activation1(gamma) * 2
        beta = self.activation2(beta)
        return gamma, beta

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
