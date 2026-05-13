from datasets import transform_size, device
import yaml
from log import logger
import os

cfg = {
    "shapley":{
        "patch": 32,
        "sample_times": 1,
        "n_per_batch": 1,
        "static_center": False,
    },
    "advGAN": {
        "adv_lambda": 15,
        "poison_lambda": 0,
        "pert_lambda": 1,
        "det_lambda": 1,
        "gather_lambda": 0,
        "hinge_c": 0.1,
        "visibility_lambda": 0,
        "clamp_min": -0.1,
        "clamp_max": 0.1,
        "use_cuda": True,
        "models_path": './models/repeat/',
        "image_nc": 3,
        "epochs": 10,
        "BOX_MIN": 0,
        "BOX_MAX": 1,
        # 三个不同阶段的generator和discriminator的lr
        "G_lr1": 0.01,
        "G_lr2": 0.001,
        "G_lr3": 0.0001,
        "D_lr1": 0.01,
        "D_lr2": 0.001,
        "D_lr3": 0.0001,
        "iou_thres": 0.5,
        "bound": 0.05, # the bound of visibility_loss
    },
    "super_aug": [1, 10, 5, 1],
    "epoch": 20,
    'lr': 0.01,
    'bound': 0.1,
    "image_size": transform_size,
    "device": device,
    "detector_path": "./frequency_detector.pt",
    "visible_trigger": "./visible_trigger.pt",
    "yolov5_path": "./infected_yolov5.pt",
    "clean_path": "./clean_yolov5.pt",
    "sigma_bound": 5.0
}

with open("class.yaml", 'r', encoding="UTF-8") as stream:
    try:
        cfg['advGAN']['classes'] = yaml.safe_load(stream)
        logger.info(f"config: {cfg}")
    except yaml.YAMLError as exc:
        logger.error(exc)
        
cnt = 1
test_models_path = cfg['advGAN']['models_path']
while os.path.exists(test_models_path):
    test_models_path = cfg['advGAN']['models_path'].strip().lstrip(os.sep) + str(cnt)
    cnt += 1
cfg['advGAN']['models_path'] = test_models_path
os.makedirs(cfg['advGAN']['models_path'])
