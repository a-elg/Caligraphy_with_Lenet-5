# Caligraphy_with_Lenet-5
A Lenet-5 like CNN to read characters, part of a bigger project called PALALA (Will be referenced soon)

### Necesitarás:
- python 3.10.7
- pip
- OpenCV
- pyzbar
- pytorch
- torchvision
- CUDA 11.7 (opcional)
  - CuDNN (opcional)

Para instalar OpenCV, pyzbar, pytorch y torchvision, puedes usar pip:
```
$ pip install opencv-python
$ pip install pyzbar
$ pip install torch torchvision torchaudio
```

Si se desea entrenar en una GPU, se debe instalar la versión de pytorch para CUDA y cuDNN, más información en [pytorch.org](https://pytorch.org/get-started/locally/)
La versión ocupada aquí es CUDA 11.7, se puede instalar pythorch integrado con CUDA con:
```
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```
Para más información sobre la instalación de CUDA y cuDNN, se puede consultar la [documentación oficial](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) (windows)

### Utilización:
Para entrenar la red, se debe ejecutar el archivo train.py, el cual tiene los siguientes argumentos:
