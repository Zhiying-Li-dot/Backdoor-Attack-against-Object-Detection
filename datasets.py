import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from log import logger


def handle_label_file(size, filename):
    # read content from labels and return numpy array
    label_info = []
    with open(filename, "r", encoding="UTF-8") as f:
        for line in f.readlines():
            line = line.strip()
            line_parts = line.split(" ")
            box_class = float(line_parts[0])  # class type
            bbox = line_parts[1:]
            x = float(bbox[0]) * size[0]  # central x
            w = float(bbox[2]) * size[0]  # bbox width
            y = float(bbox[1]) * size[1]  # central y
            h = float(bbox[3]) * size[1]  # bbox height
            label_info.append([box_class, x, y, w, h])
    return torch.tensor(label_info).to(device)


class CustomImageDataset(Dataset):
    def __init__(self, image_dir, label_dir=None, transform_function=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        if label_dir is not None:
            self.label_files = [f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))]
        self.transform = transform_function
        self.n_images = len(self.image_files)

    def __len__(self):
        self.n_images = len(self.image_files)
        return self.n_images

    def __getitem__(self, idx):
        global device
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        if self.label_dir is not None and idx < len(self.label_files):
            label_path = os.path.join(self.label_dir, self.label_files[idx])
        image = Image.open(image_path).convert('RGB')
        image_size = image.size
        if self.transform:
            image = self.transform(image)
        # image = image.permute(1, 2, 0)
        # return numpy array
        # so that every batch got in iterations will be numpy arrays
        image_array = image.to(device)
        if self.label_dir is not None and idx < len(self.label_files):
            label_array = handle_label_file(image_size, label_path)
        else:
            # all clean image set to 0
            label_array = torch.zeros_like(image_array, device=device)
        return image_array, label_array, torch.tensor(image_size), os.path.basename(image_path)
def custom_collate(batch):
    images = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]  # 由于label_array形状不同，我们只是将它们放在列表中
    image_sizes = torch.stack([item[2] for item in batch])
    image_paths = [item[3] for item in batch]
    return images, labels, image_sizes, image_paths

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# image path
# train_image_dir = "/data/zyli/datasets/coco/images/train/chosen"
# train_image_dir = "/data/zyli/datasets/coco/images/val/all"
# val_image_dir = "/data/zyli/datasets/coco/images/val/all"
# val_label_dir = "/data/zyli/datasets/coco/labels/val/all"
# test_image_dir = "/data/zyli/datasets/coco/images/test/all"
test_image_dir = "./origin"
# test_image_dir = "/data/zyli/datasets/coco/images/test/chosen"
# invisible_test_image_dir = "/data/zyli/generated_datasets/smooth_poisoned_0.2_very_invisible/images/test/"
invisible_test_image_dir = "./invisible"
# invisible_test_image_dir = "/data/zyli/generated_datasets/smooth_poisoned_0.2_very_invisible/images/chosen"
batch_size = 32
# transform_size = (128, 128)
transform_size = (640, 640)
# transform function
transform = transforms.Compose([
    transforms.Resize(transform_size),
    transforms.ToTensor(),
    ])
# transform = transforms.Compose([
#     transforms.Resize(transform_size),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# # train
# train_dataset = CustomImageDataset(image_dir=train_image_dir, transform_function=transform)
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
# train_length = len(train_dataset)
# # val
# val_dataset = CustomImageDataset(image_dir=val_image_dir, transform_function=transform, label_dir=None)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
# val_length = len(val_dataset)

# batch_size = 1
# test
test_dataset = CustomImageDataset(image_dir=test_image_dir, transform_function=transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
test_length = len(test_dataset)
# invisible poisoned test
invisible_test_dataset = CustomImageDataset(image_dir=invisible_test_image_dir, transform_function=transform)
invisible_test_dataloader = DataLoader(invisible_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
invisible_test_length = len(invisible_test_dataset)
# logger.info(f"train_length: {train_length}, val_length: {val_length}, test_length: {test_length}")
logger.info(f"test_length: {test_length}")
