import os

from datasets import test_dataloader, test_length, device, invisible_test_dataloader, invisible_test_length
from datasets import transform_size as image_size
from log import logger
from loss import TriggerLoss
import torch
from torch import nn
from torch_dct import dct_2d, idct_2d
from tqdm import tqdm
from infected_model import infected_predict, detect_and_visualize
import wandb
from painter import plot_3d_heatmaps
from torchvision import transforms
from advGAN_train import AdvGAN_Attack
from config import cfg
from gaussian_generator import TotalGenerator
import warnings

# torch.cuda.empty_cache()

# def invisible_poison(image):
#     global invisible_trigger
#     image = dct_2d((image * 255.0), 'ortho') + invisible_trigger
#     image = torch.clamp((idct_2d(image, 'ortho') / 255.0), 0, 1)
#     return image

# def visible_poison(image, visible_trigger):
#     return torch.clamp((image + visible_trigger), 0, 1)

# logger.info(f"config: {cfg}")

# # 生成可见trigger的模型
# logger.info("initializing visible trigger model...")
# visible_model = TotalGenerator()
# optimizer = torch.optim.Adadelta(visible_model.parameters(), cfg['lr'])
# criterion = TriggerLoss(cfg['super_aug'], cfg['bound'])
# visible_model = visible_model.to(device)
wandb.init(project="visible_trigger", config=cfg, resume="allow", save_code=True)
warnings.filterwarnings('ignore')
# wandb.watch(visible_model)

# 加载不可见trigger
# logger.info("loading invisible trigger...")
# invisible_trigger_path = cfg['invisible_trigger_path']
# invisible_trigger = torch.load(invisible_trigger_path).to(device)

# logger.info("into the training cycle")
# min_loss = 10000
# for n_epoch in tqdm(range(cfg['epoch'])):
#     all_losses = []
#     sample_x = None
#     iter_num = 0
#     for sample, invisible_sample in tqdm(zip(test_dataloader, invisible_test_dataloader), total=-(-test_length // test_dataloader.batch_size)):
#     # for sample, invisible_sample in tqdm(zip(test_dataloader, invisible_test_dataloader), total=1):
#         image, label, sz = sample
#         invisible_x, _, _ = invisible_sample
#         # 生成可见trigger
#         visible_x = visible_model(image)
#         # wandb.log({"Gaussian maxium": torch.max(visible_x)})
#         # visible_x = torch.zeros_like(image)
#         # invisible_x = torch.zeros_like(image)
#         # 用可见trigger和不可见trigger毒化
#         # for i in range(image.shape[0]):
#         #     visible_x[i] = visible_poison(image[i], visible_trigger)
#             # invisible_x[i] = invisible_poison(image[i])
#         # tmp = infected_predict(image)
#         # print(tmp)
#         # 计算loss
#         loss = criterion.compute_multiple_loss(visible_x, invisible_x, image, label)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         wandb.log({
#             "visibility_loss": torch.mean(criterion.losses[0]),
#             # "shapley_loss": torch.mean(criterion.losses[1]),
#             "decision_loss": torch.mean(criterion.losses[2]),
#             "frequency_loss": torch.mean(criterion.losses[3]),
#             "all_loss": loss.item()
#         })
#         all_losses.append(loss.item())
#         ts = transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.Resize(sz[0].tolist()[::-1]),
#             ])
#         sample_x = [wandb.Image(ts(image[0])), wandb.Image(ts(invisible_x[0])), wandb.Image(ts(visible_x[0])), wandb.Image(ts(visible_x[0] - image[0]))]
#         wandb.log({'sample_image': sample_x})
#         wandb.log({'detected sample': [detect_and_visualize(image[0], sz[0]), detect_and_visualize(invisible_x[0], sz[0]), detect_and_visualize(visible_x[0], sz[0]), ]})
#         # ({'detected sample': [detect_and_visualize(image[0], sz[0]), detect_and_visualize(invisible_x[0], sz[0]), detect_and_visualize(visible_x[0], sz[0]), ]})
#         iter_num += 1
#     mean_loss = torch.mean(torch.tensor(all_losses, device=device))
#     if mean_loss < min_loss:
#         min_loss = mean_loss
#         torch.save(visible_model.state_dict(), "best_visible_generator_model.pt")
#         # torch.save(visible_model(initial_image), f"best_visible_trigger.pt")
#     # heatmap = plot_3d_heatmaps(visible_model.clt)
#     # wandb.log({
#     #     'heatmap': wandb.Image(heatmap),
#     #     'sample_image': sample_x
#     # })

for f in os.listdir("."):
    if f.endswith(".py"):
        wandb.save(f)

advGAN = AdvGAN_Attack(
                    device,
                    3,
                    cfg['advGAN']['BOX_MIN'],
                    cfg['advGAN']['BOX_MAX'],
                    )
advGAN.train(cfg['epoch'])

wandb.finish()
