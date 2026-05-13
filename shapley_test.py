import torch
import torchvision.models.detection as detection
import torchvision.transforms as T
from PIL import Image
import numpy as np
import shap

# 加载预训练的Faster R-CNN模型
model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 加载并预处理图像
def preprocess_image(image_path):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)

image_path = "/home/swin/zyli/BackdoorAttack/PosionDiffusion/yolov5/data/images/bus.jpg"
input_tensor = preprocess_image(image_path)

# 定义一个函数来预测图像中的目标数量
def detect_objects(img_tensor):
    with torch.no_grad():
        prediction = model(img_tensor)
        # 返回检测到的目标数量
        return torch.tensor([len(prediction[0]["boxes"])])

# 使用Deep SHAP进行解释
background = torch.zeros((1, 3, 224, 224))  # 使用零背景
e = shap.DeepExplainer(model, background)
shap_values = e.shap_values(input_tensor)

# 可视化Shapley值
shap.image_plot(shap_values, np.array(input_tensor))