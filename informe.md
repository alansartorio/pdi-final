---
title: 'Rubik'
subtitle: 'Trabajo Final Procesamiento de Imágenes'
abstract: 'En este trabajo explicamos cómo conseguimos detectar los colores de una cara de un cubo de Rubik a partir de una foto mediante procesamiento de imágenes digitales.'
author:
  - '[Alan Sartorio](asartorio@itba.edu.ar)'
  - '[Lucas Gómez](lugomez@itba.edu.ar)'
date: 4/12/2024
lang: es-AR
toc: true
pdf-engine: tectonic
highlight-style: pygments
---

# Introducción

Se busca abordamos la problemática de identificar los colores de cada sticker de un cubo Rubik a partir de seis imágenes, una por cada cara del cubo.

El procesamiento de imágenes digitales nos permite segmentar, identificar bordes y clasificar colores de manera automática. Este informe detalla los métodos utilizados y los resultados obtenidos.

---

# Problema

El objetivo es procesar seis fotografías del cubo Rubik, identificar correctamente los colores de los stickers de cada cara para posteriormente reconstruir el estado del cubo. Las consideraciones realizadas a la hora de obtener las distintas imagenes fue:

- **Variaciones de iluminación:** Las imágenes pueden tener diferentes condiciones de luz, lo que afecta la percepción del color.
- **Ruido en las imágenes:** Es necesario filtrar imperfecciones que dificultan la segmentación.
- **Identificacion del color:** Se requiere segmentar las imagenes para obtener los stickers y comparar el color obtenido en cada uno.


---

# Metodología

A continuación, se describen los pasos del procesamiento de imágenes aplicado al cubo Rubik.

## 1. Captura de imágenes

Se tomaron fotografías de cada cara del cubo bajo diferentes condiciones de iluminación y posiciones. Estas variaciones en las condiciones iniciales ayudan a conocer la capacidad del algoritmo de identificar los colores.

![](output/00_chosen_input_image.png)

## 2. Detección de bordes

Para identificar los contornos de los stickers, utilizamos el algoritmo de **Canny**. Se detectan los bordes en función de los gradientes de intensidad en la imagen, permitiendo resaltar las transiciones abruptas de color.

Los pasos para identificar los bordes en la imagen son:

1. Filtrar la imagen mediante un filtro Gaussiano para suavizarla:
Antes de detectar bordes, debemos reducir el ruido en la imagen. Para esto, se utiliza un filtro Gaussiano. Este filtro aplica una convolución con una máscara dando más peso a los píxeles centrales y menos a los periféricos, suavizando las transiciones de intensidad.

2. Calcular los gradientes de intensidad y sus direcciones.

Un borde es un cambio abrupto en la intensidad de los píxeles. Para identificar estas transiciones:

- Se calcula el **gradiente de intensidad** en cada píxel. Esto se realiza mediante derivadas parciales en las direcciones \( x \) y \( y \):
  $$
  G_x = \frac{\partial I}{\partial x}, \quad G_y = \frac{\partial I}{\partial y}
  $$

- La magnitud del gradiente (\( G \)) indica la fuerza del cambio de intensidad:
  $$
  G = \sqrt{G_x^2 + G_y^2}
  $$

- La dirección del gradiente indica hacia dónde ocurre el cambio más pronunciado:
  $$
  \theta = \arctan\left(\frac{G_y}{G_x}\right)
  $$

![](output/01_edge_detection.png)

## 3. Erosión

La erosión reduce el tamaño de los objetos en una imagen, eliminando píxeles en sus bordes. Este proceso ayuda a eliminar el ruido y a separar elementos que puedan estar conectados en la imagen procesada. Al aplicar la erosión, se obtiene una representación más limpia y definida de las características relevantes, por lo que facilita la posterior deteccion de las caras del cubo.

![](output/02_erode.png)

## 4. Detección de contornos

La **detección de contornos** permite identificar los límites de las caras del cubo. Para este proceso, utilizamos un algoritmo de detección de contornos que segmenta la imagen y facilita la identificación de formas geométricas relevantes.

Para dicho proceso, se realizan los siguientes pasos:

1. **Encontrar los contornos**: Usamos el algoritmo `cv2.findContours` para detectar los contornos de la imagen procesada. Esto nos proporciona una lista de contornos, junto con su jerarquía.
      
    ```python
    contours, hierarchy = cv2.findContours(
        edges,
        cv2.RETR_TREE, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    ```

![](output/03_contours.png)


2. **Filtrado de contornos**: Se filtran los contornos para identificar aquellos que corresponden a formas relevantes, como cuadrados o grupos de cuadrados. Este filtrado se realiza con base en varios criterios:

- **Área de los contornos**: Se calcula el área de cada contorno y la suma de los contornos hijos. Si el área total supera un umbral determinado, el contorno es descartado, ya que se asume que pertenece a una forma sólida.

- **Contornos en los bordes**: Si un contorno toca el borde de la imagen, se descarta, ya que no se considera relevante para la detección de la cara del cubo.

- **Aproximación de polígonos**: Para simplificar los contornos, se utiliza el método `cv2.approxPolyDP`, que aproxima los contornos a una forma poligonal. Si el polígono tiene menos de 4 vértices o si los ángulos no son rectos, el contorno es descartado.

    ```python
    epsilon = 0.05 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    ```

![](output/04_categorized_contours.png)

3. **Filtrado por área válida**: Después de aproximar los contornos a polígonos, se filtran aquellos cuyo área no se ajusta a un rango válido. Si el área de un contorno es demasiado pequeña o demasiado grande, el contorno es descartado.



## 5. Rellenado y dilatación de contornos
Después de filtrar los contornos, se realiza una operación morfológica de cierre para mejorar la continuidad de los contornos entre los stickers. Esta operación consiste en aplicar primero una dilatación y luego una erosión, lo que ayuda a cerrar los posibles huecos entre los contornos de las superficies.

    ```python
    filtered_contours = cv2.morphologyEx(
        filtered_contours,
        cv2.MORPH_CLOSE, 
        utils.ones_circle(31)
    )
    ```

## 6. Rellenado y dilatación de contornos

Después de haber filtrado los contornos, se utiliza la función `getPerspectiveTransform` para calcular la matriz de transformación que ajusta la perspectiva de la cara del cubo.

    ```python
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    rectified_face = cv2.warpPerspective(img, M, (width, height))
    ```
    
![](output/05_filled_contours.png)

Con la dilatación se amplían los bordes detectados, conectando discontinuidades.

![](output/06_dilate.png)

## 7. Rectificación de caras

Mediante la transformación de perspectiva (`getPerspectiveTransform`), se alineó la cara del cubo a un plano regular. Esto permite dividirla en nueve regiones, donde cada una corresponde a un sticker.

![](output/07_find_quad.png)
![](output/rectified_face.png)

---

# Clasificación de colores

Para cada región, calculamos la media de color y comparamos con una base de datos de colores predefinidos (rojo, azul, verde, amarillo, blanco y naranja). Se utilizó el modelo de color **HSV**, que es menos sensible a variaciones de iluminación que el modelo RGB.

---

# Resultados

Los colores detectados para cada cara del cubo Rubik se muestran a continuación:

![](output/output_images.png)

Los métodos aplicados lograron identificar correctamente los colores de cada sticker, incluso bajo condiciones de iluminación no ideales.

---

# Conclusiones

Este trabajo demuestra que el procesamiento de imágenes es una herramienta poderosa para resolver problemas de recocimiento de patrones en un cubo Rubik. Las técnicas de detección de bordes, segmentación y clasificación de colores permitieron identificar con precisión los colores de cada sticker, sentando las bases para aplicaciones más complejas.

