#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 15:06:54 2023

@author: ubuntu
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 17:11:34 2023

@author: ubuntu
"""
import torch.nn as nn
import torch
import torch.nn.functional as F

class InitialLayer(nn.Module):
    def __init__(self, initialValue):
        super().__init__()
        self.xCellNumberFwd,self.yCellNumberFwd = initialValue.shape
        self.weight = torch.nn.Parameter(initialValue*torch.ones([self.xCellNumberFwd,self.yCellNumberFwd]))
    def forward(self):
        return self.weight
 
class DownsampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DownsampleLayer, self).__init__()
        self.Conv_BN_1=nn.Sequential(
          nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,padding='same',padding_mode='replicate'),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(p=0.01),
            nn.LeakyReLU(),
      )
      
        self.Conv_BN_2=nn.Sequential(
           nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=3,padding='same',padding_mode='replicate'),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(p=0.01),
            nn.LeakyReLU(),
        
      )
      
        self.downsample=nn.Sequential(
        nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=3,stride=2,padding=1),
        nn.LeakyReLU(),
        
      )

    def forward(self,x):
        
        x1=self.Conv_BN_1(x)
        x1=self.Conv_BN_2(x1)
        out_2=self.downsample(x1)
        return x1,out_2

class UpSampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(UpSampleLayer, self).__init__()
        self.Conv_BN_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch*2, kernel_size=3, padding='same',padding_mode='replicate'),
            nn.BatchNorm2d(out_ch*2),
            nn.LeakyReLU(),
           
        )
        self.Conv_BN_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch*2, out_channels=out_ch*2, kernel_size=3, padding='same',padding_mode='replicate'),
            nn.BatchNorm2d(out_ch*2),
            nn.LeakyReLU(),
            
        )
        self.Conv_BN_3=nn.Sequential(
            nn.Conv2d(in_channels=out_ch*2,out_channels=out_ch,kernel_size=3,padding='same',padding_mode='replicate'),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(),
            

        )
    def forward(self,x,out):
        
        x1=self.Conv_BN_1(x)
        x1=self.Conv_BN_2(x1)
        oo=nn.UpsamplingNearest2d(size=(out.shape[2],out.shape[3]))
        x2=oo(x1)
        x2=self.Conv_BN_3(x2)
        cat_out=torch.cat([x2,out],axis=1)
        
        return cat_out  


class SimpleUNet(nn.Module):
    def __init__(self,model_channel1,model_channel2):
        super(SimpleUNet, self).__init__()
        #out_channels=[2**(i+6) for i in range(5)] #[64, 128, 256, 512, 1024]
        out_channels=[16,32,64,128,256]
        # out_channels=[32,64,128,256,512]
        #
        self.ResTensor = InitialLayer(model_channel1)
        self.ConsTensor = InitialLayer(model_channel2)
        
        self.d1=DownsampleLayer(2,out_channels[0])
        self.d2=DownsampleLayer(out_channels[0],out_channels[1])
        self.d3=DownsampleLayer(out_channels[1],out_channels[2])
        self.d4=DownsampleLayer(out_channels[2],out_channels[3])
        #
        self.u1=UpSampleLayer(out_channels[3],out_channels[3])
        self.u2=UpSampleLayer(out_channels[4],out_channels[2])
        self.u3=UpSampleLayer(out_channels[3],out_channels[1])
        self.u4=UpSampleLayer(out_channels[2],out_channels[0])
        #
        self.o1r=nn.Sequential(
            nn.Conv2d(out_channels[1],out_channels[0],kernel_size=3,padding='same',padding_mode='replicate'),
            nn.GELU(),
        )
        self.o2r=nn.Sequential(
            nn.Conv2d(out_channels[0],2,kernel_size=3,padding='same',padding_mode='replicate'),
            nn.Tanh(),
        )
              
    def forward(self,kernal):
        a2 = self.ResTensor()[None,None,:,:]
        a3 = self.ConsTensor()[None,None,:,:]
        VInput = torch.cat([a2,a3],axis=1)
        out_1,out1=self.d1(VInput)
        out_2,out2=self.d2(out1)
        out_3,out3=self.d3(out2)
        out_4,out4=self.d4(out3)
        out5=self.u1(out4,out_4)
        out6=self.u2(out5,out_3)
        out7=self.u3(out6,out_2)
        out8=self.u4(out7,out_1)
        Vmiddel = (self.o1r(out8))
        VOutput=(self.o2r(Vmiddel))
        
        in1=torch.squeeze(VOutput[:,0,:,:])
        d1 = F.conv2d(in1[None,None,:,:],kernal,stride=[16,1])
        d1 = torch.squeeze(nn.UpsamplingNearest2d(size=(in1.shape[0],in1.shape[1]))(d1))
        
        in2=torch.squeeze(VOutput[:,1,:,:])
        d2 = F.conv2d(in2[None,None,:,:],kernal,stride=[1,16])
        d2 = torch.squeeze(nn.UpsamplingNearest2d(size=(in2.shape[0],in2.shape[1]))(d2))

        dOut1 = torch.cat([d1[None,None,:,:],d2[None,None,:,:]],axis=1)
        return  dOut1, VInput, VOutput
    
    def obtainV(self):
        a2 = self.ResTensor()[None,None,:,:]
        a3 = self.ConsTensor()[None,None,:,:]
        VInput = torch.cat([a2,a3],axis=1)
        out_1,out1=self.d1(VInput)
        out_2,out2=self.d2(out1)
        out_3,out3=self.d3(out2)
        out_4,out4=self.d4(out3)
        out5=self.u1(out4,out_4)
        out6=self.u2(out5,out_3)
        out7=self.u3(out6,out_2)
        out8=self.u4(out7,out_1)
        Vmiddel = (self.o1r(out8))
        VOutput=(self.o2r(Vmiddel))
        return out4,Vmiddel,VOutput 
    
class SimpleUNetLoss(nn.Module):
    def __init__(self):
        super(SimpleUNetLoss, self).__init__()
    def forward(self, dout, dtarget, lossf, a1, a2):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        c1 = torch.squeeze(dout[:,0,:,:])
        c2 = torch.squeeze(dout[:,1,:,:])
        tc1 = torch.squeeze(dtarget[:,0,:,:])
        tc2 = torch.squeeze(dtarget[:,1,:,:])
        l1 = lossf(c1,tc1)
        l2 = lossf(c2,tc2)
        loss = a1 * l1 + a2 * l2
        return loss, l1, l2
    
    
class PixelLayer(nn.Module):
    def __init__(self,model_channel1,model_channel2):
        super(PixelLayer, self).__init__()

        self.ResTensor = InitialLayer(model_channel1)
        self.ConsTensor = InitialLayer(model_channel2)
           
    def forward(self,kernal):
        a2 = self.ResTensor()[None,None,:,:]
        a3 = self.ConsTensor()[None,None,:,:]
        VInput = torch.cat([a2,a3],axis=1)
        
        VOutput=VInput
        
        in1=torch.squeeze(VOutput[:,0,:,:])
        d1 = F.conv2d(in1[None,None,:,:],kernal,stride=[16,1])
        d1 = torch.squeeze(nn.UpsamplingNearest2d(size=(in1.shape[0],in1.shape[1]))(d1))
       
        in2=torch.squeeze(VOutput[:,1,:,:])
        d2 = F.conv2d(in2[None,None,:,:],kernal,stride=[1,16])
        d2 = torch.squeeze(nn.UpsamplingNearest2d(size=(in2.shape[0],in2.shape[1]))(d2))

        dOut1 = torch.cat([d1[None,None,:,:],d2[None,None,:,:]],axis=1)
        return  dOut1, VInput,VOutput
    
    def obtainV(self):
        a2 = self.ResTensor()[None,None,:,:]
        a3 = self.ConsTensor()[None,None,:,:]
        VInput = torch.cat([a2,a3],axis=1)
        
        VOutput=VInput
        
        return VInput,VOutput 

class PixelCGLoss(nn.Module):
    def __init__(self):
        super(PixelCGLoss, self).__init__()
    def forward(self, VInput, dout, dtarget, lossf, a1, a2, acg):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        c1 = torch.squeeze(dout[:,0,:,:])
        c2 = torch.squeeze(dout[:,1,:,:])
        tc1 = torch.squeeze(dtarget[:,0,:,:])
        tc2 = torch.squeeze(dtarget[:,1,:,:])
        
        m1=torch.squeeze(VInput[:,0,:,:])
        m2=torch.squeeze(VInput[:,1,:,:])

        gradv_m1 = m1[ 0:-1, :] - m1[1:,:]
        gradh_m1 = m1[:,0:-1] - m1[:,1:]
        gradv_m2 = m2[ 0:-1, :] - m2[1:,:]
        gradh_m2 = m2[:,0:-1] - m2[:,1:]
        
        
        grad_m1 = torch.nn.functional.pad(torch.cat([gradv_m1[:,0:-1].reshape((-1,1)),gradh_m1[0:-1,:].reshape((-1,1))],dim=1),(0,1))
        grad_m2 = torch.nn.functional.pad(torch.cat([gradv_m2[:,0:-1].reshape((-1,1)),gradh_m2[0:-1,:].reshape((-1,1))],dim=1),(0,1))
        
        crossgradient = torch.mean(torch.abs( torch.cross(grad_m1,grad_m2)[:,2]))
        
        l1 = lossf(c1,tc1)
        l2 = lossf(c2,tc2)
        loss = a1 * l1 + a2 * l2 + acg * crossgradient

        return loss, l1, l2, crossgradient
    