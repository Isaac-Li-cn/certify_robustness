import torch

device = torch.device('cuda:0')
var = torch.rand([10, 2], device = device, requires_grad = True)

print(var)

eps = var * var

print(eps)

print(eps.unsqueeze(1))