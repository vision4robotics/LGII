import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import pytorch_colors as colors
import numpy as np
from math import exp


class L_gam(nn.Module):

    def __init__(self,patch_size):
        super(L_gam, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
    def forward(self, x ,y ):

        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        y = torch.mean(y,1,keepdim=True)
        mean = self.pool(x)
        mean_val = self.pool(y)
        mean_val = torch.pow(mean_val,0.2)
        d = torch.mean(torch.pow(mean - mean_val,2))
        return d
        
        
class L_inh(nn.Module):

    def __init__(self,patch_size,mean_val):
        super(L_inh, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    def forward(self, x , y):

        b,c,h,w = x.shape
        zero = 0.000001*torch.ones_like(x)
        y1 = torch.pow((0.16-y),3)
        y1 = 250*y1
        y1 = torch.where(y1<0.00001,zero,y1)
        x = y1*x
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val] ).cuda(),2))
        return d
        
        



# 创建一维高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x-window_size//2)**2)/float(2*sigma**2) for x in range(window_size)])
    return gauss/gauss.sum()

# 创建高斯核
def create_window(window_size, channel=1):
    # unqueeze(1) 在第二维上增加一个维度
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    # t() 转置
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# 计算ssim
# 采用归一化的高斯核来 代替计算像素的均值
def ssim(img1, img2, window, window_size, channel=3, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, stride=1, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, stride=1, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 *mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, stride=1, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, stride=1, groups=channel) - mu2_sq
    sigma_12 = F.conv2d(img1*img2, window, padding=window_size//2, stride=1, groups=channel) - mu1_mu2
    
    c1 = 0.01 **2
    c2 = 0.03 **2
    
    ssim_map = (2*sigma_12 + c2) / (sigma1_sq + sigma2_sq + c2)
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
    
class L_SSIM(torch.nn.Module):
    def __init__(self, window_size=11, channel = 3, size_average=True):
        super(L_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = create_window(window_size, channel)
        
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        
        return 1 - ssim(img1, img2, self.window, self.window_size, channel, self.size_average)