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
fieldRho = np.load('MT model/fieldRho.npy')
fieldRho = np.power(10,ToReal(fieldRho,rUpLow))

invRho = np.load('MT model/Unet/constrain1/inverted.npy')[0,0,:,:]
invRho = np.power(10,ToReal(invRho,rUpLow))

#%% SEG model
vUpLow = LowUpBound(1000,5000) # segmodel 4500 #
dUpLow = LowUpBound(1.8e3,2.6e3)

fieldVel = np.load('SEG Model/fieldVel.npy')
fieldVel = ToReal(fieldVel,vUpLow)
fieldDen = np.load('SEG Model/fieldDen.npy')
fieldDen = ToReal(fieldDen,dUpLow)

invVel = np.load('SEG Model/Unet/Joint/inverted.npy')[0,0,:,:]
invVel = ToReal(invVel,vUpLow)
invDen = np.load('SEG Model/Unet/Joint/inverted.npy')[0,1,:,:]
invDen = ToReal(invDen,dUpLow)

invDenPixel = np.load('SEG Model/Pixel/Separate GV/inverted.npy')[0,0,:,:]
Wdensity = np.load('SEG Model/Pixel/Separate GV/Wdensity.npy')
invDenPixel = ToReal(invDenPixel,dUpLow)/Wdensity

#%% Overthrust model
rUpLow = LowUpBound(0,3)
vUpLow = LowUpBound(1000,6500) 
dUpLow = LowUpBound(1.6e3,2.3e3)

invRho = np.load('Overthrust Model/Joint/inverted_2000.npy')[0,0,:,:]
invRho = ToReal(invRho,rUpLow)
invVel = np.load('Overthrust Model/Joint/inverted_2000.npy')[0,1,:,:]
invVel = ToReal(invVel,vUpLow)
invDen = np.load('Overthrust Model/Joint/inverted_2000.npy')[0,2,:,:]
invDen = ToReal(invDen,dUpLow)

#%% BP 2004 model
rUpLow = LowUpBound(0,3)
vUpLow = LowUpBound(1000,5500) 
dUpLow = LowUpBound(1.6e3,2.8e3)

invRho = np.load('BP Model/Joint/inverted.npy')[0,0,:,:]
invRho = ToReal(invRho,rUpLow)
invVel = np.load('BP Model/Joint/inverted.npy')[0,1,:,:]
invVel = ToReal(invVel,vUpLow)
invDen = np.load('BP Model/Joint/inverted.npy')[0,2,:,:]
invDen = ToReal(invDen,dUpLow)
invDenPixel = np.load('BP Model/Separate/GV/inverted.npy')[0,0,:,:]
Wdensity = np.load('BP Model/Separate/GV/Wdensity.npy')
invDenPixel = ToReal(invDenPixel,dUpLow)/Wdensity

## The Wdensity is the depth-weighting matrix for pixel-based gravity inversion.