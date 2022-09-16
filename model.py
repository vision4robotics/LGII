import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

import copy
from typing import Optional, Any

import numpy as np
import torch
from torch import nn,Tensor
import torch.nn.functional as F
import torch as t

from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout

class enhance_net_nopool(nn.Module):

    def __init__(self):
        super(enhance_net_nopool, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        
        number_f = 4
        self.e_conv0_0 = nn.Conv2d(1,number_f,3,1,1,bias=True) 
        self.e_conv1_0 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv0_1 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv2_0 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv1_1 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv0_2 = nn.Conv2d(number_f*3,number_f,3,1,1,bias=True)
        self.e_conv3_0 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
        self.e_conv2_1 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True)   
        self.e_conv1_2 = nn.Conv2d(number_f*3,number_f,3,1,1,bias=True)
        self.e_conv0_3 = nn.Conv2d(number_f*4,1,3,1,1,bias=True)
        
        self.model = nn.Sequential(

            nn.Conv2d(1, 4, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(4, affine=True),
        )
        self.model1 = nn.Sequential(

            nn.Conv2d(1, 4, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(4, affine=True),
        )

        self.F = nn.Conv2d(4, 1, 16, padding=0)
        self.q = nn.Linear(4, 4, bias=False)
        self.k = nn.Linear(4, 4, bias=False)
        self.vt = nn.Linear(4, 4, bias=False)
        self.FC = nn.Linear(4, 4, bias=False)
        self.linear1 = nn.Linear(4, 32)
        self.dropout = Dropout(0.5)
        self.linear2 = nn.Linear(32, 4)
        self.norm2 = nn.LayerNorm(4)
        self.dropout1 = Dropout(0.5)
        self.dropout2 = Dropout(0.5)
        self.dropout3 = Dropout(0.5)
         
    def forward(self, x):
        batch,w,h,b = x.shape
        red,green,blue = torch.split(x ,1,dim = 1)
        v = (red + green + blue)/3
        
        x0_0 = self.relu(self.e_conv0_0(v))
        x1_0 = self.relu(self.e_conv1_0(x0_0))
        x0_1 = self.relu(self.e_conv0_1(torch.cat([x0_0, x1_0], 1)))

        x2_0 = self.relu(self.e_conv2_0(x1_0))
        x1_1 = self.relu(self.e_conv1_1(torch.cat([x1_0, x2_0], 1)))
        x0_2 = self.relu(self.e_conv0_2(torch.cat([x0_0, x0_1, x1_1], 1)))
        
        x3_0 = self.relu(self.e_conv3_0(x2_0))
        x2_1 = self.relu(self.e_conv2_1(torch.cat([x2_0, x3_0], 1)))
        x1_2 = self.relu(self.e_conv1_2(torch.cat([x1_0, x1_1, x2_1], 1)))
        v_r = F.sigmoid(self.e_conv0_3(torch.cat([x0_0, x0_1, x0_2, x1_2], 1)))
        

        
        r = v_r

        v32 = F.interpolate(v,size=(32, 32),mode='nearest')
        q = F.interpolate(r,size=(32, 32),mode='nearest')
        v32 = self.model(v32)
        q = self.model1(q)
        v32 = v32.view(batch, 4, 256).permute(2, 0, 1)
        q = q.view(batch, 4, 256).permute(2, 0, 1)       
        k = self.k(v32)
        vt = self.vt(v32)
        q = self.q(q)
        k = k.permute(1,2,0).view(batch,4,16,16)
        q = q.permute(1,2,0).view(batch,4,16,16)
        vt = vt.permute(1,2,0).view(batch,4,16,16)
        attn = k@q
        attn = attn.softmax(dim=-1)
        attn = self.dropout1(attn)
        attn = vt@attn
        attn = attn.view(batch, 4, 256).permute(2, 0, 1)
        attn = self.FC(attn)
        attn = self.dropout2(attn)
        attn2 = self.linear2(self.dropout(self.relu(self.linear1(attn))))
        attn = attn + self.dropout3(attn2)
        v3 = self.norm2(attn)
        v5 = v3.permute(1, 2, 0).view(batch, 4, 16, 16)
        level = F.sigmoid(self.F(v5)).squeeze(dim = 1).squeeze(dim = 1).squeeze(dim = 1)
        level = 0.5*level+0.3
        g = level[0].item()

        ev1 = v + (v*r)/(1-r+0.000001)
        
        v = v + 0.000001
        red1 = red/v
        green1 = green/v
        blue1 = blue/v
        red0 = red1*ev1
        green0 = green1*ev1
        blue0 = blue1*ev1
        x0 = torch.cat([red0,green0,blue0],1)
        
        zero = 0.000001*torch.ones_like(x)
        one = 0.999999*torch.ones_like(x)
        x0 = torch.where(x0>0.999999,one,x0)
        x0 = torch.where(x0<0.000001,zero,x0)
        
        for i in range(batch):
            if(i == 0):
                ev = torch.pow (torch.unsqueeze(x0[i,:,:,:],0),level[i].item())
            else:
                v1 = torch.pow (torch.unsqueeze(x0[i,:,:,:],0),level[i].item())
                ev = torch.cat([ev,v1],0)
        
        enhance_image = ev
        zero1 = torch.zeros_like(x) 
        return enhance_image,r



