import torch
import numpy as np
import torch.nn.functional as F
# from torchvision.utils import save_image
# from utils.frequency import transform_fft, transform_ifft
import os
# import time
from infected_model import infected_predict, resolve_yolov5_output
from config import cfg
from datasets import transform_size as image_size
from datasets import device
from torch_dct import dct_2d, idct_2d

cfg = cfg['shapley']
mask_size = image_size[0] // cfg['patch']

# def compute_iou(box1, box2):
#     # Calculate overlap area
#     x_overlap = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
#     y_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
#     overlap_area = x_overlap * y_overlap

#     # Calculate the area of each box
#     area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

#     # Calculate total area
#     total_area = area_box1 + area_box2 - overlap_area

#     # Calculate IoU
#     iou = overlap_area / float(total_area)
#     return iou

def iou_single(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    box1 -- first box, list or tuple with coordinates (x1, y1, x2, y2)
    box2 -- second box, list or tuple with coordinates (x1, y1, x2, y2)
    """
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou_val = inter_area / (box1_area + box2_area - inter_area)
    
    return iou_val


def compute_ious_from_multiple_boxes(boxes_101, box_single):
    """
    计算 101 幅图像中每个方框与单幅图像方框之间的 IOU 值。

    参数：
    box_101：101 幅图像的坐标元组/列表[(x1, y1, x2, y2),...]
    box_single：单张图像的坐标元组或列表（x1, y1, x2, y2）

    返回值
    ious：浮点数值列表，即 101 个方框中每个方框的 iou 值
    """

    # ious = [compute_iou(box, box_single) for box in boxes_101]
    # return ious
    max_iou_list = []

    for boxBs in boxes_101:
        ious = []
        for boxA in box_single:
            ious.extend([iou_single(boxA, boxB) for boxB in boxBs])
        if len(ious) != 0:
            max_iou_list.append(max(ious))
        else:
            max_iou_list.append(torch.tensor(0))
    return torch.tensor(max_iou_list).to(device)


def sample_mask(img_w, img_h, mask_w, mask_h, static_center=False):
    # 返回一个sampled mask, a tensor of size ((mask_w*mask_h)+1, img_w, img_h)
    length = mask_w * mask_h + 1
    order = np.random.permutation(np.arange(0, mask_w * mask_h, 1))  # Sample an order
    mask = torch.ones(length, 3, mask_w, mask_h).to(device)
    mask = mask.view(length, 3, -1)
    for j in range(1, length):
        mask[j:, :, order[j - 1]] = 0
    mask = mask.view(length, 3, mask_w, mask_h)
    if static_center:
        mask[:, :, mask_w//2, mask_h//2] = 1
    mask = F.interpolate(mask, size=[img_w, img_h],mode="nearest").float()   # 这里采用双线性插值，将mask_w * mask_h，插值成为img_w * img_h
    return mask, order

'''这里的Label是Img对应的类别，但是这里的Shapley值采用的是IoU计算的方式来进行的，因为我们的目标是使得框去除'''
def getShapley_det(img, boxes, labels, sample_times, mask_size, k=0, n_per_batch=1, static_center=False):
    # print(img.size())
    b, c, w, h = img.size()
    length = mask_size ** 2 + 1
    shap_value = torch.zeros((mask_size ** 2)).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for i in range(sample_times // n_per_batch):
                mask, order = sample_mask(w, h, mask_size, mask_size, static_center=static_center)
                mask = mask.to(device)
                base = dct_2d(img[k], 'ortho').to(device)
                base = base.expand(mask.size(0), c, w, h)
                masked_base = base * mask
                masked_img = idct_2d(masked_base, 'ortho')
                # masked_img = masked_img.cuda()
                masked_img = torch.clamp(masked_img, 0., 1.)
                # masked_img[-1] = img[k]

                # result = infected_predict(img)
                # # Get predicted boxes and their scores
                # result = infected_predict(masked_img)
                # all_pred_boxes = []
                # all_pred_scores = []
                # all_pred_labels = []
                # for idx in range(masked_img.shape[0]):
                #     # 获取每张图片的预测框，分数和标签
                #     pred_boxes = infected_predict(masked_img[idx].unsqueeze(0))[0][:, :4]
                #     # 将当前图片的预测框添加到总列表中
                #     all_pred_boxes.append(pred_boxes)

                # if len(all_pred_boxes) == 0: continue
                # Use IoU or any other metric suitable for your detection model
                # iou_values = compute_ious_from_multiple_boxes(all_pred_boxes, boxes)
                # iou_values = compute_iou(pred_boxes, boxes[k])
                # max_iou = torch.max(iou_values)
                # max_iou_baseline = iou_values[-1]  # IoU of the baseline (without mask)
                
                decision_value = resolve_yolov5_output(infected_predict(masked_img[:-1]))

                # dy = iou_values[:-1] - max_iou_baseline
                # dy = decision_value[:-1] - decision_value[-1]
                dy = decision_value
                dy[dy < 0] = 0
                if torch.any(torch.isnan(dy)):
                    raise ValueError("Nan in dy")
                shap_value[order] += dy

                # if i % 100 == 0:
                #     print(f"{i}/{sample_times}")
        shap_value /= sample_times
    return shap_value

def compute_shapley(poisoned_image, bbox, k=0):
    global mask_size
    return getShapley_det(poisoned_image, bbox[:, 1:], bbox[:, 0], cfg['sample_times'], mask_size, k, cfg['n_per_batch'], cfg['static_center'])

def compute_multiple_shapley(poisoned_images, bboxes):
    global mask_size
    ret = []
    for poisoned_image, bbox in zip(poisoned_images, bboxes):
        ret.append(getShapley_det(poisoned_image.unsqueeze(0), bbox[:, 1:].unsqueeze(0), bbox[:, 0].unsqueeze(0), cfg['sample_times'], mask_size, 0, cfg['n_per_batch'], cfg['static_center']))
    return torch.stack(ret).to(device)
