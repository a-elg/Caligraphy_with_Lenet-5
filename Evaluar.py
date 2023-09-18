from mainFuncs import Evaluar
import sys

# Leer los argumentos de la línea de comandos
# los argumentos son:
#   1. directorio del dataset (archivo imagen)
#   2. directorio del modelo (archivo pth)
#   3. directorio de los metadatos (archivo json)
#   4. eco  (True o False)

directorioEntrada=sys.argv[1]
directorioModelo=sys.argv[2]
directorioMetadata=sys.argv[3]
eco=sys.argv[4]=="True"

if eco:
    print("Directorio de entrada: "+directorioEntrada)
    print("Directorio del modelo: "+directorioModelo)
    print("Directorio de los metadatos: "+directorioMetadata)
    print("Eco: "+str(eco))
     
