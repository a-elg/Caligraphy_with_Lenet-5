import torch as t
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
import Preprocesado as pre
import torch.nn as nn
import Lenet5 as ln5
import json

clases_equivalencia=ln5.clases_equivalencia

clases=range(len(clases_equivalencia))

def EquivalenciaEnteroAClase(entero):
    return clases_equivalencia[entero]

def EquivalenciaClaseAEntero(clase):
    return clases[clases_equivalencia.index(clase)]

def Entrenar(directorioDataset,FactorEscalamiento=1,épocas=2,coeficienteAprendizaje=0.001,GPU=False,Contrastar=False,Normalizar=False,GuardarModelo=False,eco=True):
    #no cambiar
    tamañoLote=1

    if directorioDataset=="":
        print("No se ha especificado una fuente")
        return 0

    #verificar si la fuente es un directorio (si es un archivo, se cancela el proceso)
    if not os.path.isdir(directorioDataset):
        print("La fuente no es un directorio")
        return 0

    #creamos una lista con los archivos del directorio
    archivos=os.listdir(directorioDataset)

    if len(archivos)==0:
        print("El directorio está vacío")
        return 0

    #para cada archivo, se hace el preprocesado y se almacena el objeto Resultado en una lista
    dataset=[]
    #el siguiente for tiene (0,len(archivos)) porque el último valor no se toma en cuenta (es decir, len(archivos)-1
    for i in range(0,len(archivos)):
        res=pre.Preprocesar(fuente= directorioDataset+"/"+archivos[i],dimensionEntradaIA=FactorEscalamiento*32,contrastar=Contrastar,normalizar=Normalizar,eco_=eco)
        if not res.preprocesadoExitoso:
            print("El archivo "+archivos[i]+" no pudo ser preprocesado correctamente")
        else:
            for i in range(len(res.serie)):
                #el dataset se organiza en tuplas de la forma (etiqueta,imagen), donde la imagen es un tensor de 1x32x32
                dataset.append((EquivalenciaClaseAEntero(res.serie[i][0]),t.tensor(res.serie[i][1],dtype=t.float32)))
                # dataset.append((EquivalenciaClaseAEntero(res.serie[i][0]),t.tensor(res.serie[i][1],dtype=t.float32)))

    if eco:print(f"Dataset cargado:{len(dataset)} elementos ({len(dataset[0][1])}x{len(dataset[0][1][0])})")

    #nos conectamos a la GPU
    GPUDisponible=False
    procesador=t.device("cpu")
    if GPU:
        if eco:print("Conectando a GPU")
        procesador=t.device("cuda:0" if t.cuda.is_available() else "cpu")
        if procesador==t.device("cuda:0"):
            GPUDisponible=True
            if eco:print("GPU conectada")
        else:
            print("GPU no disponible")

    modelo=ln5.Lenet5(FactorEscalamiento=FactorEscalamiento).to(procesador)

    cargadorEntrenamiento=t.utils.data.DataLoader(dataset,batch_size=tamañoLote,shuffle=False)

    funcionError=nn.CrossEntropyLoss()
    functionOptimizador=t.optim.SGD(modelo.parameters(),lr=coeficienteAprendizaje)

    #se entrena la red neuronal
    '''
    pasos: la cantidad de cuadros a analizar en cada iteración (época)
    época: la cantidad de veces que se analizará todo el dataset
    tamaño del lote: la cantidad de cuadros que se analizarán en cada iteración al mismo tiempo (debe ser 1 porque se especificó que el canal de entrada es 1)
    '''
    #imprimimos una barra de progreso de 100 guiones:
    print("Proceso de entrenado:")
    
    pasos=len(cargadorEntrenamiento)
    for época in range(épocas):
        #imprimimos un cuadro de progreso en línea cuando se haya completado un múltiplo de 1% de la época
        print("|<","-"*int(época*100/épocas)," "*(100-int(época*100/épocas)),">|",end="\r")
        for i,(etiquetas,imagenes) in enumerate(cargadorEntrenamiento):
            if GPUDisponible:
                imagenes=imagenes.to(procesador)
                etiquetas=etiquetas.to(procesador)

            functionOptimizador.zero_grad()
            #se hace el forward
            salidas=modelo(imagenes)
            #se calcula el error
            error=funcionError(salidas,etiquetas)
            #se hace el backward
            error.backward()
            #se actualizan los pesos
            functionOptimizador.step()

    print("\nEntrenamiento finalizado")

    #se evalúa el modelo
    modelo.eval()
    
    with t.no_grad():
        n_correctas=0
        n_muestras=0
        n_clases_correctas=[0 for i in range(len(clases))]
        n_clases_muestras=[0 for i in range(len(clases))]

        for etiquetas,imagenes in cargadorEntrenamiento:
            if GPUDisponible:
                imagenes=imagenes.to(procesador)
                etiquetas=etiquetas.to(procesador)

            salidas=modelo(imagenes)
            _,predicciones=t.max(salidas,1)
            n_muestras+=etiquetas.size(0)
            n_correctas+=(predicciones==etiquetas).sum().item()

            for i in range(len(etiquetas)):
                etiqueta=etiquetas[i]
                prediccion=predicciones[i]
                if etiqueta==prediccion:
                    n_clases_correctas[etiqueta]+=1
                n_clases_muestras[etiqueta]+=1
       
    if eco:
        print(f'Exactitud del entrenamiento: {100*n_correctas/n_muestras}%')

        for i in range(len(clases)):
            exactitud_clase=100*n_clases_correctas[i]/n_clases_muestras[i]
            print(f'Exactitud de la clase {EquivalenciaEnteroAClase(clases[i])}: {exactitud_clase}%')

    if GuardarModelo:
        t.save(modelo.state_dict(),directorioDataset+"/modeloEntrenado.pth")
        if eco:print("Modelo guardado")
        
        #guardamos los metadatos en un objeto json
        metadatos={
            "FactorEscalamiento":FactorEscalamiento,
            "épocas":épocas,
            "coeficienteAprendizaje":coeficienteAprendizaje,
            "Contrastar":Contrastar,
            "Normalizar":Normalizar,
            "Exactitud":100*n_correctas/n_muestras
            }
        #guardamos el objeto json en un archivo
        with open(directorioDataset+"/metadatos.json","w") as archivo:
            json.dump(metadatos,archivo)
            if eco:print("Metadatos guardados")
        
    #retornamos la exactitud del entrenamiento 
    return 100*n_correctas/n_muestras

def Evaluar(directorioEntrada,directorioModelo,directorioMetadata,eco=True):
    #revisamos si los archivos existe
    if not os.path.isfile(directorioEntrada):
        print("El archivo de entrada no existe")
        return 0
    if not os.path.isfile(directorioModelo):
        print("El modelo no existe")
        return 0
    if not os.path.isfile(directorioMetadata):
        print("El archivo de metadatos no existe")
        return 0    

    #cargamos los metadatos (el archivo json)
    with open(directorioMetadata,"r") as archivo:
        metadatos=json.load(archivo)

    FacEsc=metadatos["FactorEscalamiento"]
    Con=metadatos["Contrastar"]
    Nor=metadatos["Normalizar"]

    #preprocesamos el archivo
    res=pre.Preprocesar(fuente=directorioEntrada,dimensionEntradaIA=FacEsc*32,contrastar=Con,normalizar=Nor,eco_=eco)    

    #cargamos el modelo
    modelo=ln5.Lenet5(FactorEscalamiento=FacEsc)
    modelo.load_state_dict(t.load(directorioModelo))

    #preprocesamos el archivo
    res=pre.Preprocesar(fuente=directorioEntrada)
    dataset=[]
    if not res.preprocesadoExitoso:
        print("El archivo no pudo ser preprocesado correctamente")
    else:
        #se crea el dataset
        for i in range(len(res.serie)):
            #el dataset se organiza en tuplas de la forma (etiqueta,imagen), donde la imagen es un tensor de 1x32x32
            dataset.append((EquivalenciaClaseAEntero(res.serie[i][0]),t.tensor(res.serie[i][1],dtype=t.float32)))
    


