# import torch

# device = torch.device('cuda:0')
# var1 = torch.rand([10, 2], device = device, requires_grad = True)
# var2 = torch.rand([10, 2], device = device, requires_grad = True)

# #print(var)

# # eps = torch.cat((var1, var2), 1)

# # print(eps)

# var = [var1, var2]

# print(var.len())

# #print(eps.unsqueeze(1))

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import torchvision
# import torch
# from util.dataset import load_pkl, load_mnist, load_fmnist, load_svhn
# import pickle
# from util.param_parser import ListParser

# transform = transforms.Compose([transforms.ToTensor(),
#                             transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

# data_train = datasets.MNIST(root = "./data/",
#                             transform=transform,
#                             train = True,
#                             download = True)

# data_test = datasets.MNIST(root="./data/",
#                             transform = transform,
#                             train = False)

# data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
#                                                 batch_size = 64,
#                                                 shuffle = True)

# data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
#                                             batch_size = 64,
#                                             shuffle = True)

# data_loader = load_mnist(batch_size = 10, dset = 'test', subset=None)

# data_batch, label_batch = next(data_loader)

# print(data_batch[10])

#print(data_batch)

# for file_idx, (pkl_file, label_file) in enumerate(zip((ListParser)'./mnist_bound', (ListParser)[0,1,2,3,4,5,6,7,8,9])):
#     print("file_idx:")
#     print(file_idx)
#     print("pkl_file:")
#     print(pkl_file)
#     print("label_file:")
#     print(label_file)

# df=open('./mnist_bound','rb')

# file_data=pickle.load(df)

# results = file_data['results']

# # print(results[1]['eps'])
# print(len(results[0]['eps'][0]))

import keras # 导入Keras
from keras.datasets import mnist # 从keras中导入mnist数据集
from keras.models import Sequential # 导入序贯模型
from keras.layers import Dense # 导入全连接层
from keras.optimizers import SGD # 导入优化函数