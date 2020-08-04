import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pickle
import matplotlib.pyplot as plt
from util.dataset import load_pkl, load_mnist, load_fmnist, load_svhn
from certify_add_eps import change_data

data_loader = load_mnist(batch_size = 10, dset = 'test', subset = None)
data_batch, label_batch = next(data_loader)
file_data = pickle.load(open('iter1', 'rb'))
results_old = file_data['results']
# print(results_old)
change = results_old[0]['eps']
# print(change)
data_batch = change_data(data_batch, change)
# print(data_batch)

file_data = pickle.load(open('niter2', 'rb'))
results_old = file_data['results']
# print(results_old)
change = results_old[0]['eps']
# print(change)
data_batch = change_data(data_batch, change)

file_data = pickle.load(open('niter3', 'rb'))
results_old = file_data['results']
# print(results_old)
change = results_old[0]['eps']
# print(change)
data_batch = change_data(data_batch, change)

file_data = pickle.load(open('niter4', 'rb'))
results_old = file_data['results']
# print(results_old)
change = results_old[0]['eps']
# print(change)
data_batch = change_data(data_batch, change)

file_data = pickle.load(open('niter5', 'rb'))
results_old = file_data['results']
# print(results_old)
change = results_old[0]['eps']
# print(change)
data_batch = change_data(data_batch, change)

file_data = pickle.load(open('iter6', 'rb'))
results_old = file_data['results']
# print(results_old)
change = results_old[0]['eps']
# print(change)
data_batch = change_data(data_batch, change)

file_data = pickle.load(open('iter7', 'rb'))
results_old = file_data['results']
# print(results_old)
change = results_old[0]['eps']
# print(change)
data_batch = change_data(data_batch, change)

print(data_batch[0])
fig = plt.figure()
img = data_batch[0].reshape(28, 28)
plt.imshow(img, cmap='Greys')

plt.show()

# print(data_batch[0])
# max = -1.
# m_idx = 0

# for i in range(784):
#     if max < data_batch[0][i]:
#         max = data_batch[0][i]
#         m_idx = i
# print(max)
# print(m_idx)

# file_data = pickle.load(open('iter1', 'rb'))
# results_old = file_data['results']
# # # print(results_old)
# change = results_old[0]['eps']
# # # print(change)
# # data_batch[0] = data_batch[0] + change[0]
# # # print(data_batch)

# print(change[0][355])

# file_data = pickle.load(open('iter2', 'rb'))
# results_old = file_data['results']
# # # print(results_old)
# change = results_old[0]['eps']
# # # print(change)
# # data_batch[0] = data_batch[0] + change[0]
# # # print(data_batch)

# print(change[0][355])

# file_data = pickle.load(open('iter3', 'rb'))
# results_old = file_data['results']
# # # print(results_old)
# change = results_old[0]['eps']
# # # print(change)
# # data_batch[0] = data_batch[0] + change[0]
# # # print(data_batch)

# print(change[0][355])

# file_data = pickle.load(open('iter4', 'rb'))
# results_old = file_data['results']
# # # print(results_old)
# change = results_old[0]['eps']
# # # print(change)
# # data_batch[0] = data_batch[0] + change[0]
# # # print(data_batch)

# print(change[0][355])