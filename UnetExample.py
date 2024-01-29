#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 11:09:10 2023

@author: ubuntu

"""
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function,gradcheck
from scipy.ndimage import gaussian_filter
from InvModel import *
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_channel1 = torch.zeros((128,128),dtype=torch.float32)
model_channel2 = torch.zeros((128,128),dtype=torch.float32)

#%%

matrix_size= 64
radius = matrix_size//2-8
matrix1 = torch.zeros((matrix_size,matrix_size))
center_x, center_y = matrix_size // 2, matrix_size // 2
matrix1[center_x-radius:center_x+radius,radius:matrix_size-radius]=0.6    
matrix1[radius:matrix_size-radius,center_y-radius:center_y+radius]=0.6           
matrix2 = -1 * matrix1 + 0.6

for i in range(0, 128, matrix_size):
    for j in range(0, 128, matrix_size):
        model_channel1[i:i+matrix_size, j:j+matrix_size] = matrix1
        model_channel2[i:i+matrix_size, j:j+matrix_size] = matrix1

model_channel1[64:,64:]=0
model_channel1[0:64,64:]=matrix2
model_channel2[64:,0:64]=0

#%%
gauss_channel1 = torch.from_numpy(gaussian_filter(model_channel1, sigma= 30 ))
gauss_channel2 = torch.from_numpy(gaussian_filter(model_channel2, sigma= 30 ))

model = SimpleUNet(gauss_channel1,gauss_channel2)
model = model.to(device)

    
model.ResTensor.weight.requires_grad = False
model.ConsTensor.weight.requires_grad = False

kernal = torch.ones((1,1,16,16))
kernal = kernal/kernal.sum()

stride1 = [16,1]
stride2 = [1,16]
#%%
dtar1 = F.conv2d(model_channel1[None,None,:,:],kernal,stride=stride1)
dtar1 = nn.UpsamplingNearest2d(size=(model_channel1.shape[0],model_channel1.shape[1]))(dtar1)
dtar2 = F.conv2d(model_channel2[None,None,:,:],kernal,stride=stride2)
dtar2 = nn.UpsamplingNearest2d(size=(model_channel2.shape[0],model_channel2.shape[1]))(dtar2)



plt.figure(figsize=(6,5),dpi=300)
a0=plt.imshow(torch.squeeze(model_channel1),cmap='bwr')
a0.set_clim(0,0.7)
plt.xticks([])
plt.yticks([])
plt.xlabel('X',fontsize=20)
plt.ylabel('Y',fontsize=20)
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=20)
plt.show()

plt.figure(figsize=(6,5),dpi=300)
a0=plt.imshow(torch.squeeze(model_channel2),cmap='bwr')
a0.set_clim(0,0.7)
plt.xticks([])
plt.yticks([])
plt.xlabel('X',fontsize=20)
plt.ylabel('Y',fontsize=20)
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=20)
plt.show()

plt.figure(figsize=(6,5),dpi=300)
plt.imshow(torch.squeeze(dtar1),cmap='plasma')
plt.xticks([])
plt.yticks([])
plt.xlabel('X',fontsize=20)
plt.ylabel('Y',fontsize=20)
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=20)
plt.show()

plt.figure(figsize=(6,5),dpi=300)
plt.imshow(torch.squeeze(dtar2),cmap='plasma')
plt.xticks([])
plt.yticks([])
plt.xlabel('X',fontsize=20)
plt.ylabel('Y',fontsize=20)
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=20)
plt.show()
#%%

optimizer_p = torch.optim.Adam(model.parameters(), lr = 5e-4)
scheduler_p = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_p, T_max=10000,eta_min=5e-5)

dLabel = torch.cat([dtar1,dtar2],axis=1).to(device)
floss = nn.L1Loss()
criterion = SimpleUNetLoss().to(device)
alpha1 = 1e1
alpha2 = 1e1
loss=[]
EPOCHS = 10000

for i in range(EPOCHS):
    model.train()
    optimizer_p.zero_grad()
    dout, vinput, inverted = model(kernal.to(device))
    recoverloss,l1,l2 = criterion(dout,dLabel,floss, alpha1,alpha2)
    loss.append(recoverloss.item())
    recoverloss.backward()
    optimizer_p.step()
    scheduler_p.step()
    if i % 500 == 0:
        print(recoverloss.item())
        fig, axes=plt.subplots(2,2,figsize=(10,8))
        fig.suptitle(f'Step: {i}')
        plt.subplot(2,2,1)
        a0=plt.imshow(torch.squeeze(model_channel1.to('cpu')).detach(),cmap='bwr',origin='upper')
        a0.set_clim(0,0.7)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2,2,2)
        a1=plt.imshow(torch.squeeze(inverted[:,0,:,:].to('cpu').detach()),cmap='bwr',origin='upper')
        a1.set_clim(0,0.7)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2,2,3)
        a0=plt.imshow(torch.squeeze(model_channel2.to('cpu')).detach(),cmap='bwr',origin='upper')
        a0.set_clim(0,0.7)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2,2,4)
        a1=plt.imshow(torch.squeeze(inverted[:,1,:,:].to('cpu').detach()),cmap='bwr',origin='upper')
        a1.set_clim(0,0.7)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.pause(0.01)
        plt.show()
        
    if recoverloss.item()< 1e-2:
        print(recoverloss.item())
        plt.figure(figsize=(6,5),dpi=300)
        a1=plt.imshow(torch.squeeze(inverted[:,0,:,:].to('cpu').detach()),cmap='bwr')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('X',fontsize=20)
        plt.ylabel('Y',fontsize=20)
        cbar=plt.colorbar()
        cbar.ax.tick_params(labelsize=20)
        a1.set_clim(0,0.7)
        plt.show()

        plt.figure(figsize=(6,5),dpi=300)
        a1=plt.imshow(torch.squeeze(inverted[:,1,:,:].to('cpu').detach()),cmap='bwr')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('X',fontsize=20)
        plt.ylabel('Y',fontsize=20)
        cbar=plt.colorbar()
        cbar.ax.tick_params(labelsize=20)
        a1.set_clim(0,0.7)
        plt.show()
        break

plt.figure(figsize=(6,5),dpi=300)
a1=plt.imshow(torch.squeeze(inverted[:,0,:,:].to('cpu').detach()),cmap='bwr')
plt.xticks([])
plt.yticks([])
plt.xlabel('X',fontsize=20)
plt.ylabel('Y',fontsize=20)
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=20)
a1.set_clim(0,0.7)
plt.show()

plt.figure(figsize=(6,5),dpi=300)
a1=plt.imshow(torch.squeeze(inverted[:,1,:,:].to('cpu').detach()),cmap='bwr')
plt.xticks([])
plt.yticks([])
plt.xlabel('X',fontsize=20)
plt.ylabel('Y',fontsize=20)
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=20)
a1.set_clim(0,0.7)
plt.show()        


out4,middle,inverted = model.obtainV()
middle0 = torch.squeeze(middle).to('cpu').detach()
out40 = torch.squeeze(out4).to('cpu').detach()

fig, axes=plt.subplots(4,4,figsize=(16,16))
for i in range(16):    
    plt.subplot(4,4,i+1)
    a0=plt.imshow(middle0[i,:,:],cmap='PuOr',origin='upper')
    a0.set_clim(-0.5,0.5)
    plt.xticks([])
    plt.yticks([])
fig.tight_layout()
plt.show()

fig, axes=plt.subplots(8,4,figsize=(8,16))
for i in range(32):    
    plt.subplot(8,4,i+1)
    a0=plt.imshow(out40[i,:,:],cmap='PuOr',origin='upper')
    a0.set_clim(-0.5,0.5)
    plt.xticks([])
    plt.yticks([])
fig.tight_layout()
plt.show()