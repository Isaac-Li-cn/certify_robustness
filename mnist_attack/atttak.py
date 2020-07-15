
import numpy as np
import pandas as pd
import matplotlib
from keras.datasets import mnist
from keras import backend as K
import matplotlib.pyplot as plt
from keras.models import load_model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = load_model('mnist_model.h5')
# result = model.predict_classes(x_test[0].reshape(1,784))
# print(result)

import copy
x_test_tmp = copy.deepcopy(x_test)

x_test_tmp[1191][16][15] = 255

plt.imshow(x_test_tmp[1191],cmap='gray')
plt.show()
# print(y_test[1191])
# result = model.predict_classes(x_test_tmp[1191].reshape(1,784))
# print(result)
# if result == y_test[1191]:
#     print("Yes")

# import copy
# x_test_tmp = copy.deepcopy(x_test)
# # print(len(x_test))

# count = 0
# for index in range(len(x_test)):  # 使用测试集的10000张图片进行攻击
#     result = model.predict_classes(x_test[index].reshape(1,784))
#     if result != y_test[index]:  # 首先剔除模型分类出错的图片，看看一共有多少
#         count += 1
#         continue
#     else:
#         # 每张图片28*28 = 784 个像素点，每个点的值只改成0,255两种情况
#         for i in range(28):
#             for j in range(28):
#                 for k in [0,255]:
#                     x_test_tmp[index][i][j] = k
#                     result = model.predict_classes(x_test_tmp[index].reshape(1,784))
#                     if result[0] != y_test[index]: # 如果修改一个像素点之后，模型的预测结果跟原来的预测结果不同说明攻击成功
#                         # 都打印出来看下你会发现，同一张图片该不同的地方最终预测出来的结果也不同，比如原来为3的图片，改一个像素点可能预测为5，修改另一个可能预测为6了
#                         # 根据上面的情况，单像素攻击也可以分成两类：针对性攻击和非针对性攻击
#                         print("predict:",result[0])
#                         print("true:",y_test[index])
#                         print((index,i,j,k))
#                     x_test_tmp = copy.deepcopy(x_test)
# print(count) # 285个