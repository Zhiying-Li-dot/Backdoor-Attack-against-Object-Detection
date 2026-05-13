from numpy import zeros_like
import torch
from torch import nn
import torchvision.models as models
from torch_dct import dct_2d, idct_2d
from datasets import device
from config import cfg

detector_path = cfg['detector_path']
detector = models.alexnet(pretrained=True, progress=True)
detector.classifier[6] = nn.Linear(in_features=4096, out_features=1)
detector = nn.Sequential(detector, nn.Sigmoid())
detector.load_state_dict(torch.load(detector_path))
detector = detector.to(device)
detector.eval()

def frequency_detect(image):
    global detector
    # x = torch.zeros_like(image)
    # for i in range(image.shape[0]):
    #     x[i] = dct_2d(image[i], 'ortho')
    x = dct_2d(image, 'ortho')
    return detector(x)
