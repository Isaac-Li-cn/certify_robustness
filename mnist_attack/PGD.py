import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torchattacks import PGD, FGSM, MultiAttack

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import matplotlib.pyplot as plt

mnist_train = dsets.MNIST(root='data/',
                        train=True,
                        transform=transforms.ToTensor(),
                        download=True)

mnist_test = dsets.MNIST(root='data/',
                        train=False,
                        transform=transforms.ToTensor(),
                        download=True)

image, label = mnist_test[0]
print(image.shape)

plt.imshow(image[0],cmap='gray')
plt.show()