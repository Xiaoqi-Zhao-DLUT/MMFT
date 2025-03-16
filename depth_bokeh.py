import cv2
import numpy as np
import torch
import torch.nn.functional as F


def get_log_kernel2d_pt(kernel_size, valid_ratio, normalize=False):
    """为局部区域的加权求和生成2d的核权重。

    Args:
        kernel_size (int): 核的边长。
        valid_ratio (torch.Tensor): 有效区间比例（基于边长的一半）。
        normalize (bool, optional): 是否归一化输出。

    Returns:
        torch.Tensor: 生成的核 (kernel_size*kernel_size, H, W)
    """
    assert valid_ratio.min() >= 0 and valid_ratio.max() <= 1
    ori_shape = valid_ratio.shape
    valid_ratio = -torch.log(1 - valid_ratio)

    valid_width = kernel_size // 2 * valid_ratio
    coords = torch.linspace(-(kernel_size//2), kernel_size//2, kernel_size).reshape(-1, 1, 1)

    curr_position_mask = torch.zeros_like(coords)
    curr_position_mask[kernel_size//2] = 1
    kernel_1d = torch.where(valid_width == 0, curr_position_mask, F.relu(1 - torch.abs(coords / valid_width)))
    kernel_2d = torch.einsum("xhw,yhw->xyhw", kernel_1d, kernel_1d)
    kernel_2d = kernel_2d.reshape(-1, *ori_shape)

    if normalize:
        kernel_2d = kernel_2d / kernel_2d.sum(dim=0)
    return kernel_2d


image = "" #path
depth = "" #path


image = cv2.imread(image, cv2.IMREAD_COLOR)
depth = cv2.imread(depth, cv2.IMREAD_GRAYSCALE)
depth[depth>128] = 255
depth[depth!=255] = 0
cv2.imwrite("./xxx.png", depth)
depth = 255-depth
height, width, num_channels = image.shape

depth = cv2.resize(depth, dsize=(width, height))
image = torch.from_numpy(image).float()
depth = torch.from_numpy(depth).float()
depth[depth>128] = 255
depth[depth!=255] = 0

depth = depth - depth.min()
depth = depth / depth.max()

kernel_size = 21
tgt_image = torch.zeros_like(image, dtype=torch.float32)

image = image.permute(2, 0, 1).unsqueeze(0)
image = F.pad(image, pad=(kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), mode='replicate')
image = F.unfold(image, kernel_size=kernel_size)
image = image.reshape(num_channels, kernel_size*kernel_size, height, width)
weight = get_log_kernel2d_pt(kernel_size=kernel_size, valid_ratio=depth, normalize=True)

tgt_image = (image * weight).sum(dim=1)
tgt_image = tgt_image.permute(1, 2, 0).numpy()

cv2.imwrite("./xxx.png", tgt_image.astype(np.uint8))
