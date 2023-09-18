from mainFuncs import Entrenar
import sys

# Leer los argumentos de la línea de comandos
# los argumentos son:
#   1. directorio del dataset 
#   2. factor de escalamiento (1 o 2, puede ser más grande, pero no se ha comprobado)
#   3. épocas (se recomienda arriba de 500)
#   4. coeficiente de aprendizaje (se recomienda 0.001)
#   5. GPU (True o False)
#   6. contrastar (True o False)
#   7. normalizar (True o False)
#   8. guardar modelo (True o False)
#   9. eco  (True o False)

directorioDataset=sys.argv[1]
FactorEscalamiento=int(sys.argv[2])
épocas=int(sys.argv[3])
coeficienteAprendizaje=float(sys.argv[4])
GPU=sys.argv[5]=="True"
Contrastar=sys.argv[6]=="True"
Normalizar=sys.argv[7]=="True"
GuardarModelo=sys.argv[8]=="True"
eco=sys.argv[9]=="True"

if eco:
    print("Directorio del dataset: "+directorioDataset)
    print("Factor de escalamiento: "+str(FactorEscalamiento))
    print("Épocas: "+str(épocas))
    print("Coeficiente de aprendizaje: "+str(coeficienteAprendizaje))
    print("GPU: "+str(GPU))
    print("Contrastar: "+str(Contrastar))
    print("Normalizar: "+str(Normalizar))
    print("Guardar modelo: "+str(GuardarModelo))
    print("Eco: "+str(eco))
    print(f"Precisión: {Entrenar(directorioDataset,FactorEscalamiento,épocas,coeficienteAprendizaje,GPU,Contrastar,Normalizar,GuardarModelo,eco)}%")
    print("Fin del entrenamiento")
else:
    Entrenar(directorioDataset,FactorEscalamiento,épocas,coeficienteAprendizaje,GPU,Contrastar,Normalizar,GuardarModelo,eco)
