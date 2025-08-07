import torch
a=torch.randn((32))
b=a.view(-1,1,1,1)
print(b.shape)
c=b.view(-1, *([1] * (b.dim() - 1)))
print(c.shape)