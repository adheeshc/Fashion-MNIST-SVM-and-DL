import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../fashion_mnist/utils')
import os
import time
import pickle as pkl

import mnist_reader
from init import initializeMNIST as mnist

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


import torchvision


class CNN():
	def __init__(self,mnist):
		self.mnist=mnist
		self.conv1=nn.Conv2d(in_channels=self.mnist.x_train[0].shape,out_channels=32,kernel_size=5)
		self.fc1 = nn.Linear(in_features=32, out_features=10)
    	self.fc2 = nn.Linear(in_features=10, out_features=1)

    def forward(self):
    	
    	#conv 1
    	t = self.conv1(t)
    	t = F.relu(t)
    	t = F.max_pool2d(t,kernel_size=2)

    	#fc2
    	t = self.conv2(t)
    	
    	
    	