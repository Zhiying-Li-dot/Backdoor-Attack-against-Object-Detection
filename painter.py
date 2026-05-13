import numpy as np
import matplotlib.pyplot as plt
import torch
import io

def plot_3d_heatmaps(tensors_list):
    # tensors_list的长度必须小于等于16
    x = np.linspace(0, 639, 640)
    y = np.linspace(0, 639, 640)
    x, y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(15, 15))
    for i, tensor in enumerate(tensors_list):
        ax = fig.add_subplot(4, 4, i+1, projection='3d')
        
        # z = torch.mean(tensor, dim=0).cpu().detach().numpy()
        ax.plot_surface(x, y, tensor, cmap='jet')
        if i == 0:
            ax.set_title("Original Image")
        else:
            ax.set_title(f"Layer {i}")
        
    plt.tight_layout()

    # 保存图像为字节并返回
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf
