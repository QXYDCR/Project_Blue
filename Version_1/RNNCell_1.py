import numpy as np
import torch.nn.functional as F
import torch



out = torch.Tensor([[1, 0]])
b = F.softmax(out, dim = 1)
print(b)
target = torch.LongTensor([1])
print(out.shape, target.shape)


loss = F.cross_entropy(out, target)
print(loss)

print(np.log(np.e + 1), "<-----------", -np.log(0.7311))














