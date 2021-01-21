from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

# Define Generator
class Generator(nn.Module):
    def __init__(self, ngf, nz):
        # Used to inherit the torch.nn Module
        super(Generator, self).__init__()
        # Meta Module - consists of different layers of Modules
        self.model = nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 8, 4, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, ngf, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf, 3, 4, stride=2, padding=1, bias=False),
                nn.Tanh()
        )
        
    def forward(self, input):
        output = self.model(input)
        return output
 
# Defining the discriminator
class Discriminator(nn.Module):
    def __init__(self, ndf):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
                nn.Conv2d(3, ndf, 4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=0, bias=False),
                nn.Sigmoid()
        )
        
    def forward(self, input):
        output = self.model(input)
        return output

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
