import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar
import PIL.Image as PILI
import os

#Tipo de dato que se retorna
class Resultado:
    def __init__(self, fuente):
        self.nombre = fuente.split("/")[-1].split(".")[0]
        self.origen = fuente.split(self.nombre)[0]
        #provicionalmente se guarda en la misma carpeta que la imagen original
        self.preprocesadoExitoso = False
        self.notaDeError = None
        self.serie= None

    # - Origen: ruta de la imagen a procesar
    # - PreprocesadoExitoso: indica si el preprocesado está siendo exitoso
    # - Serie: lista de imágenes que componen el resultado del preprocesado
    # - NotaDeError: indica el error que se produjo en el preprocesado (si es que hubo)

#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
# Variables globales
#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

#tamaño de la imagen (tamaño carta horizontal [216mm x 279mm]) 
altoImagen=1000 #cuidado, no debe ser muy grande, de otra forma al rotar la imagen, la línea guía se lee como un rectángulo y no como una línea, si es muy chico, los recuadros se leerán mal
anchoImagen=np.round(altoImagen*279/216,0).astype(int) #relación de aspecto de la imagen 600mm <-> 279mm
grosorMarco=np.round(altoImagen*10/216,0) #relación de aspecto del marco con respecto al alto de la imagen 2.85mm <-> 216mm
#(el 10 se obtuvo tanteando)
grosorRecuadro=np.round(grosorMarco*0.08,0).astype(int)
#(el 0.08 se obtuvo tanteando)
cantidadFilas=4 #cantidad de filas de cajas
cantidadColumnas=7 #cantidad de columnas de cajas
eco=True #indica si se imprimen mensajes de estado

def ErrorDeProcesamiento(res, nota):
    #si no se agregó una nota, se agrega una por defecto "Error de procesamiento desconocido"
    if nota is None:
        nota="Error de procesamiento desconocido"
    print(nota)
    res.preprocesadoExitoso=False
    res.notaDeError=nota
    return res

def AnalizarQR(img):
    if eco:print("Analizando QR...")

    cadena=pyzbar.decode(img)[0].data.decode('utf-8')
    if len(cadena)==0:
        return ""

    if eco:print("QR analizado")   
    return cadena

def ValidarImagen(fuente):
    if eco:print("Validando imagen...")

    #.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~
    # Leer la imagen
    #.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~

    #crear el objeto de retorno
    res=Resultado(fuente)

    #obtener la imagen original
    try:
        res_img=cv2.imread(fuente)
    except FileNotFoundError:
        return (ErrorDeProcesamiento(res, "No se pudo leer la imagen") ,None)
    except:
        return ErrorDeProcesamiento(res)

    #.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~
    #  Validad el contenido de la imagen
    #.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~

    if res_img is None:
        return (ErrorDeProcesamiento(res, "No se pudo leer la imagen"),None)
    
    if res_img.size==0 or res_img.shape[0]==0 or res_img.shape[1]==0:
        return (ErrorDeProcesamiento(res, "La imagen está vacía"),None)
    
    res.preprocesadoExitoso=True
    
    if eco:print("Imagen validada")
    return (res, res_img)

def GuardarImagen(img, dirección):
    if eco:print("Guardando imagen... (", dirección, ")")

    #revisamos que la dirección sea válida
    if dirección is None or dirección=="":
        print("Imagen no guardada, dirección no especificada")
        return False
    #revisamos si el directorio existe
    if not os.path.exists(os.path.dirname(dirección)):
        #si no existe, lo creamos
        try:
            os.makedirs(os.path.dirname(dirección))
        except:
            print("Imagen no guardada, dirección inválida")
            return False
    #guardamos la imagen (ocupamos png para evitar la compresión de las imágenes)
    try:
        cv2.imwrite(dirección+".png", img)
    except:
        print("Imagen no guardada, error al guardar")
        return False
    
    if eco:print("Imagen guardada")
    return True

def ImprimirImagen(img, nombre):
    cv2.imshow(nombre, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ReorientarImagen(res,img_original):
    if eco:print("Reorientando imagen...")
    

    #.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~
    # Manipular la imagen para facilitar la detección de los contornos
    #.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~

    #convertir la imagen a escala de grises
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

    #hacemos una copia para no afectarla con los filtros que aplicaremos
    img=img_original.copy()

    #Esta parte necesita ser revisada !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #aplicar filtro gaussiano (para reducir el ruido)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    '''
    Explicación del desenfoque gaussiano:
    1. La función cv2.GaussianBlur() toma tres argumentos: imagen de origen, tamaño del kernel y desviación estándar en la dirección X.
    2. El tamaño del kernel debe ser un entero impar positivo.
    3. La desviación estándar en la dirección X es la desviación estándar en la dirección X, y 0 significa que se determina automáticamente.
    '''

    #aplicar filtro bilateral (para reducir el ruido  y preservar los bordes)
    img = cv2.bilateralFilter(img, 20, 30, 30)

    '''
    Explicación del filtro bilateral:
    1. La función cv2.bilateralFilter() toma cinco argumentos: imagen de origen, diámetro del vecindario de píxeles, sigmaColor, sigmaSpace y borderType.
    2. El diámetro del vecindario de píxeles es el tamaño del vecindario.
    3. sigmaColor es el filtro sigma en el espacio de color.
    4. sigmaSpace es el filtro sigma en el espacio de coordenadas.
    '''
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    #aplicar canny (para facilitar la detección de los contornos)
    img = cv2.Canny(img, 10, 20)

    '''
    Explicación de Canny:
    1. La función cv2.Canny() toma tres argumentos: imagen de origen, umbral mínimo y umbral máximo.
    2. El umbral mínimo y el umbral máximo son los valores mínimo y máximo de umbral.
    3. Los píxeles con valores de gradiente más altos que el umbral máximo se consideran bordes.
    4. Los píxeles con valores de gradiente inferiores al umbral mínimo se descartan.
    5. Los píxeles con valores de gradiente entre el umbral mínimo y el umbral máximo se consideran bordes solo si están conectados a los píxeles con valores de gradiente más altos.
    '''

    #.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~
    # Detección de contornos
    #.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~

    #Detección de contornos
    contornos, jerarquia = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #si no se detectaron contornos, se retorna un error
    if len(contornos) == 0:
        return (ErrorDeProcesamiento(res, "No se detectaron contornos"),None)

    '''
    Función ilustrativa:----------------------------------------------------------------
    #imprimir los contornos detectados
    for c in contornos:
        #generamos un color por cada tupla
        color = np.random.randint(0, 255, (3)).tolist()
        #dibujar los contornos en la imagen
        for p in c:
            #dibujar un círculo en cada punto de los contornos (se varía el color para diferenciarlos)
            cv2.circle(img, tuple(p[0]), 2, color, -1)
    cv2.imshow('contornos',img)
    ---------------------------------------------------------------------------------------
    '''

    #ordenar los contornos de mayor a menor y tomar los mayores (10 en este caso)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:10]

    #encontrar el contorno más grande
    contorno = np.array([])
    areaMax=0
    for c in contornos:
        #calcular el área del contorno
        area=cv2.contourArea(c)
        #aproximar el contorno a un polígono con 4 vértices
        aproximacion=cv2.approxPolyDP(c, 0.015*cv2.arcLength(c, True), True)
        #si el área es mayor que el área máxima y el contorno tiene 4 vértices, se guarda el contorno
        if area>areaMax and len(aproximacion)==4:
            areaMax=area
            contorno=aproximacion

    #si no se detectó ningún contorno, se retorna un error
    if len(contorno) == 0:
        return (ErrorDeProcesamiento(res, "Los contornos encontrados no son válidos"),None)
    
    '''
    Función ilustrativa:----------------------------------------------------------------
    #dibujar el contorno más grande
    cv2.drawContours(img_original, [contorno], -1, (0, 255, 0), 2)
    cv2.imshow('contorno',img_original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    -------------------------------------------------------------------------------------
    '''

    #.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~
    # Encontrar esquinas
    #.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~

    #ajustar el vector de puntos del contorno a un arreglo de 4x2
    puntos=contorno.reshape(4, 2)

    #ordenar los puntos del contorno
    #los puntos se ordenan de la siguiente manera: [arriba izquierda, abajo izquierda, arriba derecha, abajo derecha]
    #la suma de las coordenadas de los puntos arriba izquierda es la mínima y abajo derecha es la mayor
    suma=puntos.sum(axis=1)
    ArIzq=puntos[np.argmin(suma)]
    AbDer=puntos[np.argmax(suma)]

    #la diferencia de las coordenadas de los puntos arriba derecha es la mínima y abajo izquierda es la mayor
    resta=np.diff(puntos, axis=1)
    ArDer=puntos[np.argmin(resta)]
    AbIzq=puntos[np.argmax(resta)]

    #calcular el ancho y alto de la imagen de salida
    anchoAbajo=np.sqrt(((AbDer[0]-AbIzq[0])**2)+((AbDer[1]-AbIzq[1])**2))
    anchoArriba=np.sqrt(((ArDer[0]-ArIzq[0])**2)+((ArDer[1]-ArIzq[1])**2))
    altoIzquierda=np.sqrt(((ArIzq[0]-AbIzq[0])**2)+((ArIzq[1]-AbIzq[1])**2))
    altoDerecha=np.sqrt(((ArDer[0]-AbDer[0])**2)+((ArDer[1]-AbDer[1])**2))

    #tamaño de la imagen de salida
    ancho=max(int(anchoAbajo),int(anchoArriba))
    alto=max(int(altoIzquierda),int(altoDerecha))

    #ajustar el tamaño de la imagen de salida con respecto a el tamaño de una hoja carta
    ''''
    Las dimensiones de una hoja carta son 215.9 cm x 279.4 mm
    Como está en horizontal, el alto es menor que el ancho.
    buscamos ajustar el alto y con ese valor ajustar el ancho
    '''
    if ancho>alto: #ya está en horizontal
        alto=np.round(altoImagen,0).astype(int)
        ancho=np.round(anchoImagen,0).astype(int)
    else: #está en vertical
        alto=np.round(anchoImagen,0).astype(int)
        ancho=np.round(altoImagen,0).astype(int)

    #(la orientación de la imagen no importa, ya que se va a rotar)

    #asignar los puntos de entrada
    puntos_entrada=np.float32([ArIzq,ArDer,AbIzq,AbDer])

    #asignar los puntos de salida
    puntos_salida=np.float32([[0,0],[ancho,0],[0,alto],[ancho,alto]])

    #.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~
    # Transformación de perspectiva y tamaño
    #.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~
    
    #obtener la matriz de transformación
    matrix=cv2.getPerspectiveTransform(puntos_entrada,puntos_salida)

    #transformar la imagen original
    img=cv2.warpPerspective(img_original,matrix,(ancho,alto))

    #recortar el marco antes detectado
    #para esto se recorta un grosorMarco de la dimensión más pequeña de la imagen	
    pixelesACortar=np.round(min(ancho,alto)/grosorMarco,0).astype(int)
    img=img[pixelesACortar:alto-pixelesACortar,pixelesACortar:ancho-pixelesACortar]
    img_original=img.copy()

    #.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~
    # Rotación de la imagen
    #.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~

    #detectar la línea guía (repetimos el proceso de detección de bordes)
    img=cv2.Canny(
            cv2.bilateralFilter(
                cv2.GaussianBlur(img,(5,5),0)
                ,20,30,30
            ),10,20
        )
    
    contornos,jerarquia=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 
    if len(contornos)==0:
        return (ErrorDeProcesamiento(res,"No se detectó la línea guía"),None)

    contornos=sorted(contornos,key=cv2.contourArea,reverse=True)[:10]
    
    contorno=np.array([])
    areaMax=0
    for c in contornos:
        area=cv2.contourArea(c)
        aproximacion=cv2.approxPolyDP(c,0.015*cv2.arcLength(c,True),True)
        if area>areaMax:
            areaMax=area
            contorno=aproximacion
    
    #si la cantidad de esquinas detectadas es 4, entonces se tiene que abstraer a una línea
    if len(contorno)==4:

        #se obtienen los puntos más cercanos a la mitad de la imagen
        centro=[ancho/2,alto/2]
        #castear a lista para poder ordenar
        contorno=contorno.tolist()
        contorno.sort(key=lambda x: np.linalg.norm(np.array(x)-np.array(centro)))

        contorno=np.array(contorno[:2])

    elif len(contorno)!=2:
        return (ErrorDeProcesamiento(res,"No se detectó la línea guía correctamente"),None)
    
    puntos=contorno.reshape(2,2)

    #con base a la posición de los puntos, detectamos la rotaciónde la imagen

    p1,p2=(puntos[0],puntos[1])

    #rotar la imagen
    #pn[0]=x, pn[1]=y
    alto=len(img)
    ancho=len(img[0])

    if p1[1] > alto/2 and p2[1] > alto/2:
        #los 2 puntos están abajo, está bien orientado
        1
    elif p1[1] < alto/2 and p2[1] < alto/2:
        #los 2 puntos están arriba, girar 180 grados
        img_original=cv2.rotate(img_original,cv2.ROTATE_180)
    elif p1[0] > ancho/2 and p2[0] > ancho/2:
        #los 2 puntos están a la derecha, girar 270 grados
        img_original=cv2.rotate(img_original,cv2.ROTATE_180)
        img_original=cv2.rotate(img_original,cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif p1[0] < ancho/2 and p2[0] < ancho/2:
        #los 2 puntos están a la izquierda, girar 90 grados
        img_original=cv2.rotate(img_original,cv2.ROTATE_90_COUNTERCLOCKWISE)
    else :
        #la orientación presenta una anomalía
        return (ErrorDeProcesamiento(res, "Anomalía en la línea guía"),None)

    pixelesACortar=np.round(pixelesACortar*2.5,0).astype(int)
    if alto>ancho:
        alto=alto+ancho
        ancho=alto-ancho
        alto=alto-ancho
    img_original=img_original[pixelesACortar:alto-pixelesACortar,pixelesACortar:ancho-pixelesACortar]
    
    
    if eco:print("Reorientación exitosa")

    return res,img_original

def SegmentarImagen(res,img,dimensionEntradaIA,contrastar=False,normalizar=False):
    if eco:print("Segmentando imagen...")

    #matriz de puntos finales (un cuadrado de 32x32 + un marco de grosorRecuadro)
    dimensionConMarco=dimensionEntradaIA+grosorRecuadro*2
    puntosFinales=np.float32([[0,0],[dimensionConMarco,0],[0,dimensionConMarco],[dimensionConMarco,dimensionConMarco]])

    #cada imagen se segmenta en 4x7 espacios iguales
    recortes=[]
    ancho=len(img[0])
    alto=len(img)
    ajusteVertical=0.95
    alto=int(alto*ajusteVertical)


    #se segmenta la imagen en 4x7 espacios iguales
    for i in range(cantidadFilas):
        for j in range(cantidadColumnas):
            #nombre del recorte
            nombreRecorte=res.nombre+"_recorte_"+str(i)+"_"+str(j)

            #se calculan las coordenadas de los recortes
            x1=int((j*ancho)/cantidadColumnas)
            x2=int(((j+1)*ancho)/cantidadColumnas)
            y1=int((i*alto)/cantidadFilas)
            y2=int(((i+1)*alto)/cantidadFilas)
            #se recorta la imagen
            recorte=img[y1:y2,x1:x2]

            #si es el último recorte, se analiza como qr
            if(i==cantidadFilas-1 and j==cantidadColumnas-1):
                qr_str=AnalizarQR(recorte)
                if(len(qr_str)<1):
                    return ErrorDeProcesamiento(res,"Error en la lectura del QR")
                else:
                    break
                
            recorte_aux=recorte.copy()

            recorte_aux=cv2.Canny(
                cv2.bilateralFilter(
                        recorte_aux,20,30,30
                ),10,20
            )

            #detectar los contornos
            contornos,jerarquia=cv2.findContours(recorte_aux,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if len(contornos)==0:
                recortes.append((nombreRecorte,False))
                continue

            #ordenar los contornos de mayor a menor
            contornos=sorted(contornos,key=cv2.contourArea,reverse=True)

            #se obtiene el contorno más grande y que sea válido
            contorno=np.array([])
            areaMax=0
            for c in contornos:
                area=cv2.contourArea(c)
                aproximacion=cv2.approxPolyDP(c,0.015*cv2.arcLength(c,True),True)
                if len(aproximacion)==4 and area>areaMax:
                    areaMax=area
                    contorno=aproximacion
            
            #si no se encontró un contorno válido, se marca como error
            if len(contorno)==0 or areaMax<=0:
                recortes.append((nombreRecorte,False))
                continue

            #se quitan posibles dimensiones adicionales
            contorno=contorno.reshape(4,2)

            #reordenar los puntos del contorno
            suma=contorno.sum(axis=1)
            ArIzq=contorno[np.argmin(suma)]
            AbDer=contorno[np.argmax(suma)]

            resta=np.diff(contorno,axis=1)
            ArDer=contorno[np.argmin(resta)]
            AbIzq=contorno[np.argmax(resta)]

            puntosEntrada=np.float32([ArIzq,ArDer,AbIzq,AbDer])

            #se obtiene la matriz de transformación
            matriz=cv2.getPerspectiveTransform(puntosEntrada,puntosFinales)

            #se aplica la transformación
            recorte_aux=cv2.warpPerspective(recorte,matriz,(dimensionConMarco,dimensionConMarco))

            #se recorta por última vez para quitar los bordes
            recorte_aux=recorte_aux[grosorRecuadro:dimensionConMarco-grosorRecuadro,grosorRecuadro:dimensionConMarco-grosorRecuadro]
            

            #se contrasta la imagen
            if contrastar:
                #vamos a aumentar el contraste haciendo que valores inferiores a 255/2 sean redondeados a 0 y los superiores a éste, a 1
                recorte_aux = np.where(recorte_aux > 255/2, 255, 0).astype(np.uint8)

            if normalizar and not contrastar:
                #hacemos que los valores de la imagen estén entre 0 y 1
                recorte_aux = recorte_aux/255

            recortes.append((recorte_aux,True))

            ImprimirImagen(recorte_aux,nombreRecorte)

            #se guarda el recorte (no se guardan porque se pasan como parámetro en memoria)
            # GuardarImagen(recorte_aux,nombreRecorte)

    #se guardan los recortes en el resultado
    res.serie=[]

    #asignamos una letra a cada recorte
    for i in range(len(qr_str)):
        #(letra,imagen(tensor))
        if recortes[i][1]: 
            res.serie.append((qr_str[i],recortes[i][0]))

    if eco:print("Segmentación exitosa")
    
    return res

def Preprocesar(fuente,dimensionEntradaIA=32,normalizar=False,contrastar=False,eco_=True):
    global eco
    eco=eco_

    if eco:print("Preprocesando ",fuente,"...")

    #leer la imagen
    res,img=ValidarImagen(fuente)

    #si la imagen es válida, se preprocesa
    if(res.preprocesadoExitoso):
        res,img=ReorientarImagen(res,img)
        if(res.preprocesadoExitoso):
            res=SegmentarImagen(res,img,dimensionEntradaIA,contrastar,normalizar)
            if(res.preprocesadoExitoso):
                if eco:print("Preprocesamiento de ",fuente," exitoso")
            else:
                print("Segmentación fallida")
        else:
            print("Reorientación fallida")
    else:
        print("Validación fallida")

    return res