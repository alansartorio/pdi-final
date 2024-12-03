---
title: 'Rubik'
subtitle: 'Trabajo Final Procesamiento de Imágenes'
author:
  - '[Alan Sartorio - 61379](asartorio@itba.edu.ar)'
  - '[Lucas Gómez - 60408](lugomez@itba.edu.ar)'
date: 4/12/2024
lang: es-AR
---

# Problema
Se desea escanear un cubo Rubik. A partir de seis fotos del cubo (una por cada cara), se identifican los colores de cada uno de los stickers. Con esta información, podremos reconstruir el cubo correctamente.

![Imágenes de entrada del cubo Rubik](output/input_images.jpg)

# Solución

## Imagen de entrada

El primer paso consiste en obtener la imagen del cubo Rubik. En este caso, se ha tomado una fotografía de cada cara del cubo.  
Entre las distintas pruebas realizadas, se ha variado la iluminación del cubo y la posición del mismo, así como también los colores a identificar.

![Ejemplo de imagen de entrada](output/00_chosen_input_image.png)

## Detección de bordes
Se comienza encontrando los bordes de la imagen. Para esto, se utilizó el algoritmo de Canny.  
![](output/01_edge_detection.png)

## Erosión
Una vez identificados los contornos, se procede a aplicar una erosión para eliminar el ruido de los bordes detectados y aumentar la separación entre las superficies.  
El objetivo de este procedimiento es facilitar la tarea de detección de la cara del cubo.  
![](output/02_erode.png)

## Contornos
Se detectan los contornos de la imagen. Este proceso se realiza para identificar los límites de la cara, para posteriormente segmentarla e identificar los colores.

![](output/03_contours.png)

## Categorización de contornos
Los contornos obtenidos son filtrados para identificar formas geométricas que sean relevantes para la detección del cubo. Se utilizaron los siguientes criterios:

| **Criterio**            | **Descripción**                                                                 |
|-------------------------|---------------------------------------------------------------------------------|
| **Área acumulada**       | Si la suma del área de los contornos hijos supera un umbral, el contorno se descarta.   |
| **Contornos en bordes**  | Si el contorno toca el borde, se descarta.                                      |
| **Cantidad de vértices** | Si tiene menos de 4 vértices, se descarta.                                      |
| **Ángulos no rectos**    | Si algún ángulo no es recto, el contorno se descarta.                          |

## Categorización de contornos
![](output/04_categorized_contours.png)

## Rellenado de contornos
Se procede a rellenar los contornos válidos.  
![](output/05_filled_contours.png)

## Dilatación de contornos
Mediante este proceso, se mejora la continuidad de los contornos mediante el uso de una operación de cierre:

1. **Dilatación**
2. **Erosión**

Finalmente, se obtiene la silueta aproximada de la cara del cubo.
![](output/06_dilate.png)

## Contornos

Se identifican los contornos que corresponden a la cara del cubo mediante el siguiente proceso:

1. **Búsqueda de contornos**: Se detectan los contornos en la imagen procesada.
2. **Filtrado de contornos**:
   - **Aproximación poligonal**: Los contornos se aproximan a formas simples (rectángulos).
   - **Descartar contornos no rectangulares**: Solo se conservan los contornos con 4 vértices.
   - **Filtrado por área**: Se eliminan los contornos fuera de un rango de área válido.
3. **Selección del contorno más grande**: Elegimos el contorno con el área más grande, que corresponde a la cara del cubo.

## Contornos
![](output/07_find_quad.png)

## Obtención de colores
Teniendo el contorno de la cara, se procede a dividir el área en nueve y encontrar el color que más se asemeje al de cada sección.

# Resultados
Finalmente, los resultados encontrados fueron los siguientes:

![](output/output_images.png)
