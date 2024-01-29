#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 17:43:46 2024

@author: ubuntu
"""
import numpy as np
import torch 
import scipy
class LowUpBound:
  def __init__(self, low,up):
    self.low = low
    self.up = up
def ToNormal(a, vUpLow,range_min = -1, range_max = 1):
    return (range_max-range_min)* (a-vUpLow.low) / (vUpLow.up - vUpLow.low) + range_min
def ToReal(a,vUpLow, range_min = -1, range_max = 1):
    return (a - range_min) * (vUpLow.up - vUpLow.low) / (range_max-range_min) + vUpLow.low  

#%% MT model
rUpLow = LowUpBound(0,3)
fieldRho = np.load('MT_Model/fieldRho.npy')
fieldRho = np.power(10,ToReal(fieldRho,rUpLow))

invRho = np.load('MT_Model/Unet/constrain1/inverted.npy')[0,0,:,:]
invRho = np.power(10,ToReal(invRho,rUpLow))

#%% SEG model
vUpLow = LowUpBound(1000,5000) # segmodel 4500 #
dUpLow = LowUpBound(1.8e3,2.6e3)

fieldVel = np.load('SEG_Model/fieldVel.npy')
fieldVel = ToReal(fieldVel,vUpLow)
fieldDen = np.load('SEG_Model/fieldDen.npy')
fieldDen = ToReal(fieldDen,dUpLow)

invVel = np.load('SEG_Model/Unet/Joint/inverted.npy')[0,0,:,:]
invVel = ToReal(invVel,vUpLow)
invDen = np.load('SEG_Model/Unet/Joint/inverted.npy')[0,1,:,:]
invDen = ToReal(invDen,dUpLow)

invDenPixel = np.load('SEG_Model/Pixel/Separate_GV/inverted.npy')[0,0,:,:]
Wdensity = np.load('SEG_Model/Pixel/Separate_GV/Wdensity.npy')
invDenPixel = ToReal(invDenPixel,dUpLow)/Wdensity

#%% Overthrust model
rUpLow = LowUpBound(0,3)
vUpLow = LowUpBound(1000,6500) 
dUpLow = LowUpBound(1.6e3,2.3e3)

invRho = np.load('Overthrust_Model/Joint/inverted_2000.npy')[0,0,:,:]
invRho = ToReal(invRho,rUpLow)
invVel = np.load('Overthrust_Model/Joint/inverted_2000.npy')[0,1,:,:]
invVel = ToReal(invVel,vUpLow)
invDen = np.load('Overthrust_Model/Joint/inverted_2000.npy')[0,2,:,:]
invDen = ToReal(invDen,dUpLow)

#%% BP 2004 model
rUpLow = LowUpBound(0,3)
vUpLow = LowUpBound(1000,5500) 
dUpLow = LowUpBound(1.6e3,2.8e3)

invRho = np.load('BP_Model/Joint/inverted.npy')[0,0,:,:]
invRho = ToReal(invRho,rUpLow)
invVel = np.load('BP_Model/Joint/inverted.npy')[0,1,:,:]
invVel = ToReal(invVel,vUpLow)
invDen = np.load('BP_Model/Joint/inverted.npy')[0,2,:,:]
invDen = ToReal(invDen,dUpLow)
invDenPixel = np.load('BP_Model/Separate/GV/inverted.npy')[0,0,:,:]
Wdensity = np.load('BP_Model/Separate/GV/Wdensity.npy')
invDenPixel = ToReal(invDenPixel,dUpLow)/Wdensity

## The Wdensity is the depth-weighting matrix for pixel-based gravity inversion.