# Índice
- Preprocesado
- - Escalado de grises
- - Observaciones


# Preprocesado
Para que el procesado de la imagen sea más rápido y eficiente, se aplican una serie de transformaciones a la imagen antes de ser procesada por la red neuronal.

## Escalado de grises
Las razones detrás del escalado de grises es la reducciónd de la dimensionalidad de la imagen y la eliminación de información innecesaria para el entrenamiento de la red neuronal.
La fórmula para convertir una imagen a escala de grises es la siguiente:
```
Y' = 0.299 R + 0.587 G + 0.114 B
```
Donde `R`, `G` y `B` son los valores de los canales rojo, verde y azul respectivamente.
La razón por la que se usan esos valores es porque el ojo humano es más sensible al color verde que al rojo y al azul, por lo que se le da más peso a la información del canal verde. (Basado en el modelo estándar de luminancia de la ITU-R BT.709)
Más información en [OpenCV](https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html)

## Observaciones
Para obtener mejores resultados se sugiere que la fotografía sea tomada con una buena iluminación y que el fondo sea lo más uniforme posible, es decir, sin muchos detalles que puedan confundir al algoritmo (como pinturas, manteles, etc.)
También es importante que el marco principal de la imagen se pueda visualizar en su totalidad, de otra forma, el algoritmo no podrá detectar el marco (o lo hará incorrectamente) y no podrá procesar la imagen o los resultados pueden ser muy malos.
Es importante mencionar que para un resultado correcto, es preciso tomar la foto lo más perpendicular a la foto posible.
