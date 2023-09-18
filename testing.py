#colocamos un registro de tiempo para medir el tiempo de ejecución
import time
from mainFuncs import Entrenar

def Entrenamiento24():
    inicio=time.time()
    res=[]

    for éps in range (2):
        for coefs in range (2):
            for fes in range (2):
                for tipo in range(3):
                    if tipo==0:
                        norm=False
                        cont=False
                    elif tipo==1:
                        norm=True
                        cont=False
                    elif tipo==2:
                        norm=False
                        cont=True

                    inicio_iteracion=time.time()
                    ép=int(500*(éps+1))
                    coef=10**(-(2+coefs))
                    faes=fes+1

                    presi=Entrenar(directorioDataset="./Dataset",FactorEscalamiento=faes,épocas=ép,coeficienteAprendizaje=coef,GPU=True,Normalizar=norm,Contrastar=cont,GuardarModelo=False,eco=False)
                    
                    fin_iteracion=time.time()
                    tiempo_iteracion=fin_iteracion-inicio_iteracion
                    horas_iteracion=int(tiempo_iteracion/3600)
                    minutos_iteracion=int((tiempo_iteracion-horas_iteracion*3600)/60)
                    segundos_iteracion=int(tiempo_iteracion-horas_iteracion*3600-minutos_iteracion*60)

                    res.append((ép,coef,faes,norm,cont,presi,horas_iteracion,minutos_iteracion,segundos_iteracion))
                    print(f'Ép:{ép}|CA:{coef}|FE:{faes}|N:{norm}|C:{cont}|P:{presi}|T:{horas_iteracion}-{minutos_iteracion}-{segundos_iteracion}')


    #ahora se para el tiempo
    fin=time.time()
    tiempo=fin-inicio
    horas=int(tiempo/3600)
    minutos=int((tiempo-horas*3600)/60)
    segundos=int(tiempo-horas*3600-minutos*60)

    print(f'Tiempo de ejecución: {horas} horas, {minutos} minutos, {segundos} segundos')

    for i in range(len(res)):
        print(f'Ép:{res[i][0]}|CA:{res[i][1]}|FE:{res[i][2]}|N:{res[i][3]}|C:{res[i][4]}|P:{res[i][5]}|T:{res[i][6]}-{res[i][7]}-{res[i][8]}')

def Entrenamiento24Filtrado():
    inicio=time.time()
    res=[]
    '''
    Ép:500  |FE:2 |N:T |C:F |P:100.0 |T:0-12-8
    Ép:1000 |FE:1 |N:T |C:F |P:100.0 |T:0-20-4
    Ép:1000 |FE:2 |N:T |C:F |P:100.0 |T:0-26-34
    Ép:500  |FE:1 |N:F |C:T |P:100.0 |T:0-9-1
    Ép:500  |FE:2 |N:F |C:T |P:100.0 |T:0-12-17
    Ép:1000 |FE:1 |N:F |C:T |P:100.0 |T:0-18-8
    Ép:1000 |FE:2 |N:F |C:T |P:100.0 |T:0-24-20
    Ép:500  |FE:1 |N:F |C:F |P:100.0 |T:0-9-3
    Ép:500  |FE:2 |N:F |C:F |P:100.0 |T:0-12-0
    Ép:1000 |FE:1 |N:F |C:F |P:100.0 |T:0-18-8
    Ép:1000 |FE:2 |N:F |C:F |P:100.0 |T:0-24-26

    
    Entrenar(directorioDataset="./Dataset",FactorEscalamiento=2,épocas=500 ,coeficienteAprendizaje=10**(-2),GPU=True,Normalizar=True ,Contrastar=False,GuardarModelo=False,eco=False)
    Entrenar(directorioDataset="./Dataset",FactorEscalamiento=1,épocas=1000,coeficienteAprendizaje=10**(-2),GPU=True,Normalizar=True ,Contrastar=False,GuardarModelo=False,eco=False)
    Entrenar(directorioDataset="./Dataset",FactorEscalamiento=2,épocas=1000,coeficienteAprendizaje=10**(-2),GPU=True,Normalizar=True ,Contrastar=False,GuardarModelo=False,eco=False)
    Entrenar(directorioDataset="./Dataset",FactorEscalamiento=1,épocas=500 ,coeficienteAprendizaje=10**(-2),GPU=True,Normalizar=False,Contrastar=True ,GuardarModelo=False,eco=False)
    Entrenar(directorioDataset="./Dataset",FactorEscalamiento=2,épocas=500 ,coeficienteAprendizaje=10**(-2),GPU=True,Normalizar=False,Contrastar=True ,GuardarModelo=False,eco=False)
    Entrenar(directorioDataset="./Dataset",FactorEscalamiento=1,épocas=1000,coeficienteAprendizaje=10**(-2),GPU=True,Normalizar=False,Contrastar=True ,GuardarModelo=False,eco=False)
    Entrenar(directorioDataset="./Dataset",FactorEscalamiento=2,épocas=1000,coeficienteAprendizaje=10**(-2),GPU=True,Normalizar=False,Contrastar=True ,GuardarModelo=False,eco=False)
    Entrenar(directorioDataset="./Dataset",FactorEscalamiento=1,épocas=500 ,coeficienteAprendizaje=10**(-2),GPU=True,Normalizar=False,Contrastar=False,GuardarModelo=False,eco=False)
    Entrenar(directorioDataset="./Dataset",FactorEscalamiento=2,épocas=500 ,coeficienteAprendizaje=10**(-2),GPU=True,Normalizar=False,Contrastar=False,GuardarModelo=False,eco=False)
    Entrenar(directorioDataset="./Dataset",FactorEscalamiento=1,épocas=1000,coeficienteAprendizaje=10**(-2),GPU=True,Normalizar=False,Contrastar=False,GuardarModelo=False,eco=False)
    Entrenar(directorioDataset="./Dataset",FactorEscalamiento=2,épocas=1000,coeficienteAprendizaje=10**(-2),GPU=True,Normalizar=False,Contrastar=False,GuardarModelo=False,eco=False)

    '''
    #ahora se para el tiempo
    ultimo=inicio

    print(f"Ép:500 |FE:2|N:T|C:F|P:{Entrenar(directorioDataset='./Dataset',FactorEscalamiento=2,épocas=500 ,coeficienteAprendizaje=10**(-2),GPU=True,Normalizar=True ,Contrastar=False,GuardarModelo=False,eco=False)}|T:",end="")
    tiempo_iteracion=time.time()
    aux=tiempo_iteracion-ultimo
    print(f"{int(aux/3600)}-{int((aux-int(aux/3600)*3600)/60)}-{int(aux-int(aux/3600)*3600)-int((aux-int(aux/3600)*3600)/60)*60}")
    ultimo=tiempo_iteracion
    
    print(f"Ép:1000|FE:1|N:T|C:F|P:{Entrenar(directorioDataset='./Dataset',FactorEscalamiento=1,épocas=1000,coeficienteAprendizaje=10**(-2),GPU=True,Normalizar=True ,Contrastar=False,GuardarModelo=False,eco=False)}|T:",end="")
    tiempo_iteracion=time.time()
    aux=tiempo_iteracion-ultimo
    print(f"{int(aux/3600)}-{int((aux-int(aux/3600)*3600)/60)}-{int(aux-int(aux/3600)*3600)-int((aux-int(aux/3600)*3600)/60)*60}")
    ultimo=tiempo_iteracion
    
    print(f"Ép:1000|FE:2|N:T|C:F|P:{Entrenar(directorioDataset='./Dataset',FactorEscalamiento=2,épocas=1000,coeficienteAprendizaje=10**(-2),GPU=True,Normalizar=True ,Contrastar=False,GuardarModelo=False,eco=False)}|T:",end="")
    tiempo_iteracion=time.time()
    aux=tiempo_iteracion-ultimo
    print(f"{int(aux/3600)}-{int((aux-int(aux/3600)*3600)/60)}-{int(aux-int(aux/3600)*3600)-int((aux-int(aux/3600)*3600)/60)*60}")
    ultimo=tiempo_iteracion
    
    print(f"Ép:500 |FE:1|N:F|C:T|P:{Entrenar(directorioDataset='./Dataset',FactorEscalamiento=1,épocas=500 ,coeficienteAprendizaje=10**(-2),GPU=True,Normalizar=False,Contrastar=True ,GuardarModelo=False,eco=False)}|T:",end="")
    tiempo_iteracion=time.time()
    aux=tiempo_iteracion-ultimo
    print(f"{int(aux/3600)}-{int((aux-int(aux/3600)*3600)/60)}-{int(aux-int(aux/3600)*3600)-int((aux-int(aux/3600)*3600)/60)*60}")
    ultimo=tiempo_iteracion
    
    print(f"Ép:500 |FE:2|N:F|C:T|P:{Entrenar(directorioDataset='./Dataset',FactorEscalamiento=2,épocas=500 ,coeficienteAprendizaje=10**(-2),GPU=True,Normalizar=False,Contrastar=True ,GuardarModelo=False,eco=False)}|T:",end="")
    tiempo_iteracion=time.time()
    aux=tiempo_iteracion-ultimo
    print(f"{int(aux/3600)}-{int((aux-int(aux/3600)*3600)/60)}-{int(aux-int(aux/3600)*3600)-int((aux-int(aux/3600)*3600)/60)*60}")
    ultimo=tiempo_iteracion
    
    print(f"Ép:1000|FE:1|N:F|C:T|P:{Entrenar(directorioDataset='./Dataset',FactorEscalamiento=1,épocas=1000,coeficienteAprendizaje=10**(-2),GPU=True,Normalizar=False,Contrastar=True ,GuardarModelo=False,eco=False)}|T:",end="")
    tiempo_iteracion=time.time()
    aux=tiempo_iteracion-ultimo
    print(f"{int(aux/3600)}-{int((aux-int(aux/3600)*3600)/60)}-{int(aux-int(aux/3600)*3600)-int((aux-int(aux/3600)*3600)/60)*60}")
    ultimo=tiempo_iteracion
    
    print(f"Ép:1000|FE:2|N:F|C:T|P:{Entrenar(directorioDataset='./Dataset',FactorEscalamiento=2,épocas=1000,coeficienteAprendizaje=10**(-2),GPU=True,Normalizar=False,Contrastar=True ,GuardarModelo=False,eco=False)}|T:",end="")
    tiempo_iteracion=time.time()
    aux=tiempo_iteracion-ultimo
    print(f"{int(aux/3600)}-{int((aux-int(aux/3600)*3600)/60)}-{int(aux-int(aux/3600)*3600)-int((aux-int(aux/3600)*3600)/60)*60}")
    ultimo=tiempo_iteracion
    
    print(f"Ép:500 |FE:1|N:F|C:F|P:{Entrenar(directorioDataset='./Dataset',FactorEscalamiento=1,épocas=500 ,coeficienteAprendizaje=10**(-2),GPU=True,Normalizar=False,Contrastar=False,GuardarModelo=False,eco=False)}|T:",end="")
    tiempo_iteracion=time.time()
    aux=tiempo_iteracion-ultimo
    print(f"{int(aux/3600)}-{int((aux-int(aux/3600)*3600)/60)}-{int(aux-int(aux/3600)*3600)-int((aux-int(aux/3600)*3600)/60)*60}")
    ultimo=tiempo_iteracion
    
    print(f"Ép:500 |FE:2|N:F|C:F|P:{Entrenar(directorioDataset='./Dataset',FactorEscalamiento=2,épocas=500 ,coeficienteAprendizaje=10**(-2),GPU=True,Normalizar=False,Contrastar=False,GuardarModelo=False,eco=False)}|T:",end="")
    tiempo_iteracion=time.time()
    aux=tiempo_iteracion-ultimo
    print(f"{int(aux/3600)}-{int((aux-int(aux/3600)*3600)/60)}-{int(aux-int(aux/3600)*3600)-int((aux-int(aux/3600)*3600)/60)*60}")
    ultimo=tiempo_iteracion
    
    print(f"Ép:1000|FE:1|N:F|C:F|P:{Entrenar(directorioDataset='./Dataset',FactorEscalamiento=1,épocas=1000,coeficienteAprendizaje=10**(-2),GPU=True,Normalizar=False,Contrastar=False,GuardarModelo=False,eco=False)}|T:",end="")
    tiempo_iteracion=time.time()
    aux=tiempo_iteracion-ultimo
    print(f"{int(aux/3600)}-{int((aux-int(aux/3600)*3600)/60)}-{int(aux-int(aux/3600)*3600)-int((aux-int(aux/3600)*3600)/60)*60}")
    ultimo=tiempo_iteracion
    
    print(f"Ép:1000|FE:2|N:F|C:F|P:{Entrenar(directorioDataset='./Dataset',FactorEscalamiento=2,épocas=1000,coeficienteAprendizaje=10**(-2),GPU=True,Normalizar=False,Contrastar=False,GuardarModelo=False,eco=False)}|T:",end="")
    tiempo_iteracion=time.time()
    aux=tiempo_iteracion-ultimo
    print(f"{int(aux/3600)}-{int((aux-int(aux/3600)*3600)/60)}-{int(aux-int(aux/3600)*3600)-int((aux-int(aux/3600)*3600)/60)*60}")
    ultimo=tiempo_iteracion


    fin=time.time()
    tiempo=fin-inicio
    horas=int(tiempo/3600)
    minutos=int((tiempo-horas*3600)/60)
    segundos=int(tiempo-horas*3600-minutos*60)

    print(f'Tiempo de ejecución: {horas} horas, {minutos} minutos, {segundos} segundos')

def EntrenamientoS():
    Entrenar

EntrenamientoS()

