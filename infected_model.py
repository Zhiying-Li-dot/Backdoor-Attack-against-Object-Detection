import torch
import yolov5
from datasets import device
from datasets import transform_size as image_size
from torchvision import transforms
from PIL import Image, ImageDraw
import wandb
from config import cfg
# 加载自定义模型
# yolov5_model = yolov5.load(cfg['yolov5_path']).cuda()
yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom', path=cfg['yolov5_path'], device=device)
# 确保模型处于评估模式
# yolov5_model.to(device)
yolov5_model.eval()
# set model parameters
# yolov5_model.conf = 0.5  # NMS confidence threshold
# yolov5_model.iou = 0.45  # NMS IoU threshold
# yolov5_model.agnostic = False  # NMS class-agnostic
# yolov5_model.multi_label = False  # NMS multiple labels per box
# yolov5_model.max_det = 1000  # maximum number of detections per image

clean_model = torch.hub.load('ultralytics/yolov5', 'custom', path=cfg['clean_path'], device=device)
clean_model.eval()

def infected_predict(image):
    result = yolov5_model([(img * 255.0).cpu().detach().numpy() for img in image])
    # result.print()
    return result.pred

def clean_predict(image):
    result = clean_model([(img * 255.0).cpu().detach().numpy() for img in image])
    return result.pred

def infected_predict_PIL(image):
    result = yolov5_model([img for img in image])
    # result.print()
    return result.pred

def infected_predict_by_path(image):
    result = []
    for img in image:
        img.save("tmp.jpg")
        result.append(yolov5_model("tmp.jpg", size=640).pred[0])
        # yolov5.detect.run(source="tmp.jpg", weights=yolov5_path, conf_thres=0.5, imgsz=640)
    # result.print()
    return result

def transform_output(model_output):
    trans_result = []

    for output in model_output:
        if output.size(0) == 0:
            continue

        # 取前4列
        x1, y1, x2, y2 = output[:, 0], output[:, 1], output[:, 2], output[:, 3]

        # 归一化宽度和高度
        ws = (x2 - x1) / image_size[0]
        hs = (y2 - y1) / image_size[1]

        xs = x1 / image_size[0] + ws / 2
        ys = y1 / image_size[1] + hs / 2

        # 合并结果
        trans_coords = torch.stack([xs, ys, ws, hs], dim=1).to(device)
        trans_result.append(trans_coords)

    return trans_result

def resolve_yolov5_output(model_output):
    # 只针对消失攻击
    # 使用切片操作将坐标、置信度和类别分开
    # 使用torch.empty初始化S
    if len(model_output) == 0:
        return torch.zeros_like(model_output)
    S = torch.empty(len(model_output), device=device)
    for idx, output in enumerate(model_output):
        if len(output) == 0:
            S[idx] = 0
            continue
        coords = output[:, :4]  # 取前4列
        confs = output[:, 4].unsqueeze(-1)   # 取第5列并增加一个维度以方便后续操作
        # 归一化宽度和高度
        ws = (coords[:, 2] - coords[:, 0]) / image_size[0]
        hs = (coords[:, 3] - coords[:, 1]) / image_size[1]
        # S[idx] = torch.max(ws * hs * confs)
        S[idx] = (torch.exp(ws * hs * confs) - 1).mean()
        # S[idx] = (torch.exp(ws * hs) - 1).mean()
    S = torch.where(torch.isnan(S) | torch.isinf(S), torch.zeros_like(S), S)
    return S



def detect_and_visualize(tensor_img, sz, caption="Detected Image"):
    """
    Args:
    - tensor_img (torch.Tensor): 输入的图像 Tensor，大小为 [3, 640, 640]
    - sz (torch.Size): 原图的大小
    - results: YOLOv5 的检测结果
    
    Returns:
    - wandb Image 对象
    """
    # 创建一个 torchvision transforms pipeline，将 Tensor 转为 PIL Image 并调整大小
    transformer = transforms.Compose([
        transforms.ToPILImage(),
    ])
    
    # 将输入的图像 Tensor 转换为 PIL Image
    img_pil = transformer(tensor_img)
    results = infected_predict([tensor_img])[0]
    if len(results) == 0:
        return wandb.Image(transforms.Resize(sz.tolist()[::-1])(img_pil), caption=caption)
    boxes = results[:, :4]
    confidences = results[:, 4]
    labels = results[:, 5]
    
    # 在 PIL Image 上绘制边界框
    draw = ImageDraw.Draw(img_pil)
    for label, conf, box in zip(labels, confidences, boxes):
        if conf > 0.5: # 只绘制置信度大于0.5的边界框
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            # 这里假设您有模型的名称，以便将标签名称绘制到图像上
            draw.text((x1, y1 - 10), f"Class {int(label)} {conf:.2f}", fill="red")
    # import uuid
    # img_pil.save(f"{uuid.uuid4()}.png")
    img_pil = transforms.Resize(sz.tolist()[::-1])(img_pil)
    # 转换为wandb Image 对象
    return wandb.Image(img_pil, caption=caption)

def clean_detect_and_visualize(tensor_img, sz, caption="Detected Image"):
    """
    Args:
    - tensor_img (torch.Tensor): 输入的图像 Tensor，大小为 [3, 640, 640]
    - sz (torch.Size): 原图的大小
    - results: YOLOv5 的检测结果
    
    Returns:
    - wandb Image 对象
    """
    # 创建一个 torchvision transforms pipeline，将 Tensor 转为 PIL Image 并调整大小
    transformer = transforms.Compose([
        transforms.ToPILImage(),
    ])
    
    # 将输入的图像 Tensor 转换为 PIL Image
    img_pil = transformer(tensor_img)
    results = clean_predict([tensor_img])[0]
    if len(results) == 0:
        return wandb.Image(transforms.Resize(sz.tolist()[::-1])(img_pil), caption=caption)
    boxes = results[:, :4]
    confidences = results[:, 4]
    labels = results[:, 5]
    
    # 在 PIL Image 上绘制边界框
    draw = ImageDraw.Draw(img_pil)
    for label, conf, box in zip(labels, confidences, boxes):
        if conf > 0.5: # 只绘制置信度大于0.5的边界框
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            # 这里假设您有模型的名称，以便将标签名称绘制到图像上
            draw.text((x1, y1 - 10), f"Class {int(label)} {conf:.2f}", fill="red")
    # import uuid
    # img_pil.save(f"{uuid.uuid4()}.png")
    img_pil = transforms.Resize(sz.tolist()[::-1])(img_pil)
    # 转换为wandb Image 对象
    return wandb.Image(img_pil, caption=caption)

def draw_boxes(tensor, boxes):
    """
    根据给定的框将张量中的对应区域的值设置为1。

    参数:
    - tensor: 要修改的张量，形状为 CxHxW。
    - boxes: 一个list，其中的元素为 (x,y,w,h) 形式的框。

    返回:
    - 修改后的张量。
    """
    for box in boxes:
        x, y, w, h = box
        tensor[:, y:y+h, x:x+w] = 1
    return tensor

def draw_multi_boxes(tensors, boxes_list):
    ret = torch.zeros_like(tensors)
    for i in range(tensors.shape[0]):
        ret[i] = draw_boxes(tensors[i], boxes_list[i])
    return ret

def make_mask(tensors, rate = 0.1):
    model_output = infected_predict(tensors)
    trans_output = transform_output(model_output)
    for idx, item in enumerate(trans_output):
        item[:, 2] *= rate  # 对ws进行处理
        item[:, 3] *= rate  # 对hs进行处理
        trans_output[idx] = item
    return draw_multi_boxes(tensors, trans_output)
