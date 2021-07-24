import torch

from Utils import *

# SEED = 1234
# torch.manual_seed(SEED)
#
# input = torch.randn(3, 3)
# print(input)
#
# aa = nn.Softmax(dim = 1)
# a = aa(input)
# print(a)
#
# b = torch.log(a)
# print(b)
#
# nll = nn.NLLLoss()
# target = torch.tensor([0, 1, 1])
# mark_star()
# print(nll(b, target))
#
# print("{:.2f} + {:.2f} + {:.2f} = {:.2f}".format(-b[0][0], -b[1][1], -b[2][1], (b[0][0]+b[1][1]+b[2][1]) / 3))
#
#
# mark_star()
# cc = nn.CrossEntropyLoss()
# print(cc(input, target))


a = torch.randn(1, 2)
b = torch.randn(1, 2)

c = torch.randn(2, 1)
print(c, c.shape)
print(c.squeeze(), c.squeeze().shape)