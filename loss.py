# 引入必要的库和模块
import torch
from torch import nn
from compute_shapley import compute_shapley, compute_multiple_shapley
from infected_model import infected_predict, resolve_yolov5_output
from frequency_detector import frequency_detect
from datasets import transform_size as image_size
from datasets import device
import torch.nn.functional as F

# 定义可见性损失，其目的是度量两个输入之间的MSE（均方误差）与给定的bound之间的MSE
class VisiblilityLoss(nn.Module):
    def __init__(self, bound=0.5):
        super(VisiblilityLoss, self).__init__()
        self.mse = nn.MSELoss()
        # self.mse = nn.SmoothL1Loss()
        self.bound = torch.tensor(bound).to(device)

    def forward(self, y_pred, y_true):
        return self.mse(self.mse(y_pred, y_true), self.bound)
        # return self.mse(y_pred, y_true)
        # return 1/self.mse(y_pred, y_true)

# 定义Shapley损失，其目的是度量两个输入shapley值之间的MSE
class ShapleyLoss(nn.Module):
    def __init__(self):
        super(ShapleyLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, visible_shapley, invisible_shapley):
        return self.mse(visible_shapley, invisible_shapley)

# 定义决策损失，其目的是处理Yolov5模型的输出并计算其平均值
class DecisionLoss(nn.Module):
    def __init__(self):
        super(DecisionLoss, self).__init__()
        # self.bce = nn.BCELoss()
        # self.mse = nn.MSELoss()

    def forward(self, model_output):
        # result = resolve_yolov5_output(model_output).mean()
        # result = F.sigmoid(result)
        # return self.bce(result, torch.zeros_like(result))
        return resolve_yolov5_output(model_output).mean()
    
class PoisonLoss(nn.Module):
    def __init__(self):
        super(PoisonLoss, self).__init__()
        self.decision = DecisionLoss()

    def forward(self, infected_model_output, clean_model_output):
        return torch.exp(10*(self.decision(infected_model_output) - self.decision(clean_model_output)))

# 定义频率损失，其目的是度量输入和全0张量之间的二进制交叉熵
class FrequencyLoss(nn.Module):
    def __init__(self):
        super(FrequencyLoss, self).__init__()
        # self.bce = nn.BCELoss()
        # self.mse = nn.MSELoss()
        self.logloss = lambda x: -torch.log(1 - x)

    def forward(self, y_pred):
        return self.logloss(y_pred).mean()
    
class GatherLoss(nn.Module):
    def __init__(self):
        super(GatherLoss, self).__init__()
        self.logloss = lambda x: -torch.log(1 - x)
        def compute_entropy(img):
            # 将3xHxW的RGB图像转换为1xHxW的灰度图像
            img_gray = img.mean(dim=0, keepdim=True)
            # 量化像素值，例如，将[0,1]分为256个bin
            img_quantized = (img_gray * 255).int()
            # 计算每个像素值的频率
            p = torch.histc(img_quantized.float(), bins=256, min=0, max=255) / img_quantized.numel()
            # 计算熵
            entropy = -(p * torch.log2(p + 1e-9)).sum()
            return entropy
        def compute_multi_entropy(imgs):
            return torch.tensor([compute_entropy(img) for img in imgs], device=device).mean()
        self.entropy = lambda x: (torch.exp(compute_multi_entropy(x))-1).mean()
        # self.entropy = compute_multi_entropy

    def forward(self, perturbations):
        return self.entropy(perturbations)

# 定义总的触发器损失，其目的是将上述定义的所有损失函数组合起来
class TriggerLoss(nn.Module):
    def __init__(self, super_aug, bound=0.5):
        super(TriggerLoss, self).__init__()
        self.visibility_loss = VisiblilityLoss(bound)
        self.shapley_loss = ShapleyLoss()
        self.decision_loss = DecisionLoss()
        self.frequency_loss = FrequencyLoss()
        self.super_aug = super_aug
        self.losses = []

    # 计算批量图像的损失，并最后返回平均损失
    def forward(self, visible_poisoned_images, invisible_poisoned_images, origin_images, bboxes):
        batch_size = len(visible_poisoned_images)
        individual_losses = torch.empty(batch_size, device=device)  # 为losses预先分配空间
        for idx, (visible_poisoned_image, invisible_poisoned_image, origin_image, bbox) in enumerate(zip(visible_poisoned_images, invisible_poisoned_images, origin_images, bboxes)):
            loss = self.compute_single_loss(visible_poisoned_image.unsqueeze(0), invisible_poisoned_image.unsqueeze(0), origin_image.unsqueeze(0), bbox, 0)
            individual_losses[idx] = loss  # 直接在tensor上操作
        return torch.mean(individual_losses)  # 返回的是一个1D tensor

    # 计算单个图像的总损失，其中各部分的损失由各自的权重调节
    def compute_single_loss(self, visible_poisoned_image, invisible_poisoned_image, origin_image, bbox, k=0):
        total_loss = self.super_aug[0] * self.visibility_loss(visible_poisoned_image, origin_image)
        total_loss += self.super_aug[1] * self.shapley_loss(compute_shapley(visible_poisoned_image, bbox, k), compute_shapley(invisible_poisoned_image, bbox, k))
        total_loss += self.super_aug[2] * self.decision_loss(infected_predict(visible_poisoned_image))
        total_loss += self.super_aug[3] * self.frequency_loss(frequency_detect(visible_poisoned_image))
        return total_loss
    
    # 计算单个图像的总损失，其中各部分的损失由各自的权重调节
    def compute_multiple_loss(self, visible_poisoned_images, invisible_poisoned_images, origin_images, bboxes):
        self.losses = []
        self.losses.append(self.visibility_loss(visible_poisoned_images, origin_images))
        total_loss = self.super_aug[0] * self.losses[0]
        # self.losses.append(self.shapley_loss(compute_multiple_shapley(visible_poisoned_images, bboxes), compute_multiple_shapley(invisible_poisoned_images, bboxes)))
        # total_loss += self.super_aug[1] * self.losses[1]
        self.losses.append(torch.tensor(0, device=device))
        self.losses.append(self.decision_loss(infected_predict(visible_poisoned_images)))
        total_loss += self.super_aug[2] * self.losses[2]
        self.losses.append(self.frequency_loss(frequency_detect(visible_poisoned_images)))
        total_loss += self.super_aug[3] * self.losses[3]
        return total_loss

'''定义了像素惩罚界'''
def pixel_penalty(original, generated, lower_threshold, upper_threshold):
    mse = F.mse_loss(original, generated, reduction='none').mean(dim=(1,2,3))  # 计算像素级别的MSE
    penalty = torch.where(mse < lower_threshold, lower_threshold - mse, torch.zeros_like(mse))  # 计算低于下限阈值的惩罚
    penalty += torch.where(mse > upper_threshold, mse - upper_threshold, torch.zeros_like(mse))  # 计算超过上限阈值的惩罚
    return penalty.mean()  # 返回平均惩罚值
