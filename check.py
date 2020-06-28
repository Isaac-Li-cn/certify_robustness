import torch

device = torch.device('cuda:0')
var1 = torch.rand([10, 2], device = device, requires_grad = True)
var2 = torch.rand([10, 2], device = device, requires_grad = True)

#print(var)

eps = torch.cat((var1, var2), 1)

print(eps)

#print(eps.unsqueeze(1))