import torch
from datasets import test_dataloader, test_length
from datasets import transform_size as image_size
import os
from tqdm import tqdm
from advGAN_model import Generator
from torch_dct import dct_2d, idct_2d
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = "models/frequencynetG_best.pth"
model = Generator(3, 3).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

num = 0
save_folder = "poisoned_image/"
os.makedirs(save_folder, exist_ok=True)

with torch.no_grad():
    for images, label, sz, name in tqdm(test_dataloader, total=-(-test_length // test_dataloader.batch_size)):
        fre_images = dct_2d(images, 'ortho')
        pert = model(fre_images)
        new_fre_images = fre_images + pert
        new_images = torch.clamp(idct_2d(new_fre_images, 'ortho'), 0, 1)
        for i in range(images.shape[0]):
            ts = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(sz[i].tolist()[::-1]),
            ])
            pil_image = ts(new_images[i])
            pil_image.save(os.path.join(save_folder, name[i]))
            num += 1

print("DONE.")
