import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import Myloss
import numpy as np
from torchvision import transforms


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)





def train(config):

    os.environ['CUDA_VISIBLE_DEVICES']='0'

    LGII_net = model.enhance_net_nopool().cuda()

    LGII_net.apply(weights_init)
    if config.load_pretrain == True:
        LGII_net.load_state_dict(torch.load(config.pretrain_dir))
    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)     
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)



    L_gam = Myloss.L_gam(16)
    L_inh = Myloss.L_inh(16,0)
    L_SSIM = Myloss.L_SSIM()


    optimizer = torch.optim.Adam(LGII_net.parameters(), lr=config.lr,weight_decay=config.weight_decay)
    
    LGII_net.train()

    for epoch in range(config.num_epochs):
        for iteration, img_lowlight in enumerate(train_loader):

            img_lowlight = img_lowlight.cuda()

            enhanced_image,A  = LGII_net(img_lowlight)

            loss_gam = 20*torch.mean(L_exp(enhanced_image, img_lowlight))
            loss_inh = 40*torch.mean(L_inh(enhanced_image, img_lowlight))
            loss_ssim = L_SSIM(enhanced_image, img_lowlight)
            
            # best_loss
            loss =  loss_ssim  + loss_gam + loss_inh
            #

            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(LGII_net.parameters(),config.grad_clip_norm)
            optimizer.step()

            if ((iteration+1) % config.display_iter) == 0:
                print("Loss at iteration", iteration+1, ":", loss.item())
            if ((iteration+1) % config.snapshot_iter) == 0:
                
                torch.save(LGII_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')       




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--load_pretrain', type=bool, default= False)
    parser.add_argument('--pretrain_dir', type=str, default= "snapshots/Epoch49.pth")

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)


    train(config)








    
