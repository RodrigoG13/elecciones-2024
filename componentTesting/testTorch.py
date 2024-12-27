import torch

# Crea un tensor en la GPU
x = torch.rand(10000, 10000).cuda()
y = torch.mm(x, x)
print(y)  # Si funciona sin errores, tu GPU está operativa.
