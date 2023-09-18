import torch
import torchvision

datos=torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

#imprimir los datos
print(datos)


