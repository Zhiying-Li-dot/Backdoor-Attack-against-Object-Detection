import os
import torch
from torch import nn
from torch_dct import dct_2d, idct_2d
from datasets import val_dataloader, val_length
from tqdm import tqdm
from torchvision import transforms
def invisible_poison(image):
    global invisible_trigger
    image = dct_2d((image * 255.0), 'ortho') + invisible_trigger
    image = torch.clamp((idct_2d(image, 'ortho') / 255.0), 0, 1)
    return image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
invisible_trigger = torch.load("/home/swin/zyli/BackdoorAttack/WeatherApproximationGAN/shapley-adv/sec1_trigger_generater/best_generator_tensor.pt").to(device)

for image, label, sz in tqdm(val_dataloader, total=-(-val_length // val_dataloader.batch_size)):
    x = invisible_poison(image[0])
    x = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(sz[0].tolist()[::-1]),
            ])(x)
    x.save("tmp.png")
    
