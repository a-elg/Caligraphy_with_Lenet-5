import numpy as np
import torch as t
import torch.nn as nn
import cv2
import Preprocesado

clases_equivalencia=[
    "A","B","C","D","E","F","G",
    "H","I","J","K","L","M","N",
    "Ñ","O","P","Q","R","S","T",
    "U","V","W","X","Y","Z",

    "a","b","c","d","e","f","g",
    "h","i","j","k","l","m","n",
    "ñ","o","p","q","r","s","t",
    "u","v","w","x","y","z"
]

clases=range(len(clases_equivalencia))

class ConvC3(nn.Module):
    def __init__(self):
        super(ConvC3, self).__init__()
        # self.kernel_size = (5,5)
        self.kernel_size = (5,5)
        # self.stride = (1,1) 
        self.stride = (1,1) 

        self.padding = (0,0) #no hay padding
        self.in_channels = 6
        self.out_channels =  16
        self.weights = nn.Parameter(t.randn(self.out_channels,self.in_channels,self.kernel_size[0],self.kernel_size[1]))
        self.bias = nn.Parameter(t.zeros(self.out_channels))

        self.C3_0  = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.C3_1  = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.C3_2  = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.C3_3  = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.C3_4  = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.C3_5  = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.C3_6  = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.C3_7  = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.C3_8  = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.C3_9  = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.C3_10 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.C3_11 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.C3_12 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.C3_13 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.C3_14 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.C3_15 = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def forward(self,x):
        X=[]
        X.append(self.C3_0(x[0:3,:,:]))
        X.append(self.C3_1(x[1:4,:,:]))
        X.append(self.C3_2(x[2:5,:,:]))
        X.append(self.C3_3(x[3:6,:,:]))
        X.append(self.C3_4(t.cat((x[4:6,:,:],x[0:1,:,:]),dim=0)))
        X.append(self.C3_5(t.cat((x[5:6,:,:],x[0:2,:,:]),dim=0)))
        X.append(self.C3_6(x[0:4,:,:]))
        X.append(self.C3_7(x[1:5,:,:]))
        X.append(self.C3_8(x[2:6,:,:]))
        X.append(self.C3_9(t.cat((x[3:6,:,:],x[0:1,:,:]),dim=0)))
        X.append(self.C3_10(t.cat((x[4:6,:,:],x[0:2,:,:]),dim=0)))
        X.append(self.C3_11(t.cat((x[5:6,:,:],x[0:3,:,:]),dim=0)))
        X.append(self.C3_12(t.cat((x[0:2,:,:],x[3:5,:,:]),dim=0)))
        X.append(self.C3_13(t.cat((x[1:3,:,:],x[4:6,:,:]),dim=0)))
        X.append(self.C3_14(t.cat((x[0:1,:,:],x[2:4,:,:],x[5:6,:,:]),dim=0)))
        X.append(self.C3_15(x))
        #son 16 canales los que tenemos como resultados en el arreglo X, se devolverá un tensor de 16x10x10
        return t.cat(X,dim=0)

class Lenet5(nn.Module):
    
    def __init__(self,TanH=False,FactorEscalamiento=1):
        super(Lenet5, self).__init__()
        self.FE=FactorEscalamiento
        #Función de activación (Tanh o ReLU)
        if(TanH):
            self.activacion = nn.Tanh()
        else:
            self.activacion = nn.ReLU()
   
        self.activacionSalida = nn.Softmax(dim=0)
        self.error=nn.CrossEntropyLoss()

        '''
        Capa 1 - Convolucional 
        La entrada es de 1 imagen de 32x32 
        El kernel es de 5x5, es decir, un filtro de convolución de 5x5
        El stride es de 1x1, es decir, el filtro se desplaza de 1 en 1
        El padding es de 0, es decir, no se añade ningún pixel alrededor de la imagen
        in_channels=1, es decir, la entrada será una imagen 
        out_channels=6, es decir, la salida trendrá 6 imágenes (cada una de ellas es una imagen de 28x28)

        Largo= L-(K1)+1 = 32-(5)+1 = 28
        '''
        self.C1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5), stride=(1,1), padding=(0,0))
        
        '''
        Capa 2 - Pooling
        La entrada es de 6 imágenes de 28x28 
        El kernel es de 2x2, es decir, un filtro de sumarización de 2x2
        El stride es de 2x2, es decir, el filtro se desplaza de 2 en 2
        El padding es de 0, es decir, no se añade ningún pixel alrededor de la imagen
        La salida es de 6 imágenes de 14x14 

        Largo= L/K2 = 28/2 = 14
        '''
        self.S2 = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=(0,0))

        '''
        Capa 3 - Convolucional
        La entrada es de 6 imágenes de 14x14 
        El kernel es de 5x5, es decir, un filtro de convolución de 5x5
        El stride es de 1x1, es decir, el filtro se desplaza de 1 en 1
        El padding es de 0, es decir, no se añade ningún pixel alrededor de la imagen
        in_channels=6, es decir, la entrada será una imagen
        out_channels=16, es decir, la salida trendrá 16 imágenes (cada una de ellas es una imagen de 10x10)

        Esta convulución en particular sigue el siguiente patrón:
                              1 1 1 1 1 1
        - 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5
        0 x       x x x     x x x x   x x
        1 x x       x x x     x x x x   x
        2 x x x       x x x     x   x x x
        3   x x x     x x x x     x   x x
        4     x x x     x x x x   x x   x
        5       x x x     x x x x   x x x

        Largo= L-(K3)+1 = 14-(5)+1 = 10

        '''
        self.C3 = ConvC3()
        # self.C3  = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), stride=(1,1), padding=(0,0))

        '''
        Capa 4 - Pooling
        La entrada es de 16 imágenes de 10x10 
        El kernel es de 2x2, es decir, un filtro de sumarización de 2x2
        El stride es de 2x2, es decir, el filtro se desplaza de 2 en 2
        El padding es de 0, es decir, no se añade ningún pixel alrededor de la imagen
        La salida es de 16 imágenes de 5x5 

        Largo= L/K4 = 10/2 = 5
        '''
        self.S4 = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=(0,0))

        '''
        Capa 5 - Convolucional
        La entrada es de 16 imágenes de 5x5 
        El kernel se calcula en la parte inferior de este comentario.
        El stride es de 1x1, es decir, el filtro se desplaza de 1 en 1
        El padding es de 0, es decir, no se añade ningún pixel alrededor de la imagen
        in_channels=16, es decir, la entrada será una imagen
        out_channels: 120
        
        El kernel de base es de 5x5, es decir, un filtro de convolución de 5x5,
        lo que hace que, cuando la entrada es de 32x32, la salida de esta capa sean
        canales de 1x1. Cuando la entrada es mayor a 32x32, la entrada de esta capa
        ya no es de 5x5, por lo que debemos adaptar el kernel para que la salida sea
        de 1x1. Para ello, se calcula el kernel de la siguiente manera:
            Si la entrada de es L
            la primera salida es de (L)-K1+1
            la segunda salida es de (L-K1+1)/K2
            la tercera salida es de ((L-K1+1)/K2)-K3+1
            la cuarta salida es de (((L-K1+1)/K2)-K3+1)/K4
            la quinta salida (y entrada de la capa 6) es de ((((L-K1+1)/K2)-K3+1)/K4)-K5+1
            
            Sustituyendo los valores de KN excepto K5, tenemos que:
            ((((L-5+1)/2)-5+1)/2)-K5+1

            Tomando en cuenta el kernell base:
            ((((L-5+1)/2)-5+1)/2)-(5+x)+1=1

            Despejando x paso a paso:
            ((((L-5+1)/2)-5+1)/2)-(5+x)+1=1
            ((((L-5+1)/2)-5+1)/2)-(5+x)=0
            ((((L-5+1)/2)-5+1)/2)=5+x
            ((((L-5+1)/2)-5+1)=10+2x
            ((((L-5+1)/2)-5)=9+2x
            (((L-5+1)/2)=14+2x
            ((L-5+1)=28+4x
            (L-5)=27+4x
            L=32+4x
            4x=L-32
            x=(L-32)/4

            Sabiendo que L es un múltiplo de 32 y que n=factor de escalamiento
            x=(32*n-32)/4
            x=(32*(n-1))/4
            x=8*(n-1)
        '''

        kernel_aux = 5 + 8*(self.FE-1)
        # self.C5 = nn.Conv2d(in_channels=16, out_channels=120*self.FE, kernel_size=(kernel_aux,kernel_aux), stride=(1,1), padding=(0,0))
        self.C5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(kernel_aux,kernel_aux), stride=(1,1), padding=(0,0))

        '''
        Capa 6 - Aplanamiento
        La entrada varía de acuerdo con la imagen de entrada inicial
        La salida es de 16*24 parámetros multiplicados por el factor de escalamiento al cuadrado (porque el factor afecta a la imagen en 2 dimensiones)
            (es el tamaño, en pixeles, máximo de una letra en un archivo de 32x32)

        '''
        # self.F6 = nn.Linear(in_features=120*self.FE, out_features=16*24*(self.FE**2))
        self.F6 = nn.Linear(in_features=120, out_features=16*24*(self.FE**2))

        '''
        Capa 7 - Salida
        La entrada es de 16*24 parámetros multiplicados por el factor de escalamiento al cuadrado (porque el factor afecta a la imagen en 2 dimensiones)
        La salida es de 54 parámetros
        '''

        self.F7 = nn.Linear(in_features=16*24*(self.FE**2), out_features=54)

    def forward(self, x):
        #Capa 1 - Convolucional
        x = self.C1(x)
        x = self.activacion(x)

        #Capa 2 - Pooling
        x = self.S2(x)

        #Capa 3 - Convolucional
        x = self.C3(x)
        x = self.activacion(x)

        #Capa 4 - Pooling
        x = self.S4(x)

        #Capa 5 - Convolucional
        x = self.C5(x)
        x = self.activacion(x)

        #Capa 6 - Aplanamiento (se toma en cuenta el tamaño del kernel, el stride y el tamaño de x)
        # x = x.view(-1, 120*self.FE)
        x = x.view(-1,120)
        x = self.F6(x)
        x = self.activacion(x)

        #Capa 7 - Salida
        x = self.F7(x)
        # x =self.activacionSalida(x)

        return x

class GenericLenet(nn.Module):
        def __init__(self):
            super(GenericLenet, self).__init__()
            self.activacion = nn.ReLU()
            self.activacionSalida = nn.Softmax()
            self.C1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5), stride=(1,1), padding=(0,0))
            self.S2 = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=(0,0))
            self.C3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), stride=(1,1), padding=(0,0))
            self.S4 = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=(0,0))
            self.C5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5,5), stride=(1,1), padding=(0,0))
            self.F6 = nn.Linear(in_features=120, out_features=84)
            self.F7 = nn.Linear(in_features=84, out_features=54)

        
        def forward(self, x):
            x = self.C1(x)
            x = self.activacion(x)
            x = self.S2(x)
            x = self.C3(x)
            x = self.activacion(x)
            x = self.S4(x)
            x = self.C5(x)
            x = self.activacion(x)
            x = x.view(-1, 120)
            x = self.F6(x)
            x = self.activacion(x)
            x = self.F7(x)
            x = self.activacionSalida(x)
            return x
