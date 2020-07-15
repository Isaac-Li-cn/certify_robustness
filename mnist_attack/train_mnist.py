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

(x_train, y_train), (x_test, y_test) = mnist.load_data() # 下载mnist数据集
# print(x_train.shape,y_train.shape) # 60000张28*28的单通道灰度图
# print(x_test.shape,y_test.shape)

import matplotlib.pyplot as plt # 导入可视化的包
# im = plt.imshow(x_train[0],cmap='gray')
# plt.show()
# y_train[0]

x_train = x_train.reshape(60000,784) # 将图片摊平，变成向量
x_test = x_test.reshape(10000,784) # 对测试集进行同样的处理
# print(x_train.shape)
# print(x_test.shape)

x_train = x_train / 255
x_test = x_test / 255
# print(x_train[0])

y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)

model = Sequential() # 构建一个空的序贯模型
# 添加神经网络层
model.add(Dense(512,activation='relu',input_shape=(784,)))
model.add(Dense(256,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()

model.compile(optimizer=SGD(),loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=64,epochs=5,validation_data=(x_test,y_test)) # 此处直接将测试集用作了验证集

score = model.evaluate(x_test,y_test)
# 保存模型
model.save('mnist_model.h5')  # creates a HDF5 file 'my_model.h5'
print("loss:",score[0])
print("accu:",score[1])