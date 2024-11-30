from typing import Optional
import cv2
import numpy as np
from cv2.typing import MatLike

from utils import Point, Quad
import utils


def find_face(img: MatLike, debug=False, extra_debug=False) -> Optional[Quad]:
    height, width = img.shape[:2]
    img_size = min(width, height)
    minimum_face_size = 0.10 * img_size
    maximum_face_size = 0.98 * img_size

    if extra_debug:
        cv2.imshow("original", img)

    # Detectar bordes
    edges = cv2.Canny(img, 100, 170)
    # Invertir para conseguir superficies
    ret, edges = cv2.threshold(255 - edges, 240, 255, cv2.THRESH_BINARY)

    if extra_debug:
        cv2.imshow("edges1", edges)
    # Eliminar piezas chicas
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, utils.ones_circle(4))

    if extra_debug:
        cv2.imshow("edges2", edges)
    # Aumentar separacion entre superficies
    edges = cv2.erode(edges, utils.ones_circle(4))

    if debug:
        cv2.imshow("edges", edges)

    contour_image = np.zeros_like(img)

    # Encontrar contornos de superficies
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    contour_image_approx = np.zeros_like(img)

    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)

    filtered_contours = np.zeros_like(edges)

    # Filtrar contornos de cuadrados o grupos de cuadrados
    hierarchy = tuple(*hierarchy)
    areas = [cv2.contourArea(contour) for contour in contours]
    for (i, contour), (next, previous, first_child, parent) in zip(
        enumerate(contours), hierarchy
    ):
        # Calcular suma de areas de contornos hijos
        children_area = 0
        next = first_child
        while next != -1:
            children_area += areas[next]
            next = hierarchy[next][0]

        # Si tiene hijos y el area supera cierto valor se descarta, se asume que los cuadrados tienen color solido
        if children_area > 0 and children_area > minimum_face_size**2:
            cv2.drawContours(contour_image_approx, [contour], -1, (255, 0, 255), 1)
            cv2.drawContours(
                contour_image_approx, contours, first_child, (0, 255, 255), 3
            )
            continue

        # Si el contorno toca un borde de la imagen se descarta
        if utils.does_contour_touch_border(contour, width, height):
            continue

        # Aproximar una figura simplificada al contorno
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Si tiene menos de 4 vertices, se descarta
        if len(approx) < 4:
            cv2.drawContours(contour_image_approx, [approx], -1, (0, 0, 255), 1)
            continue

        # Si algun angulo no es recto (o similar) se descarta
        if not utils.all_right_angles(approx):
            cv2.drawContours(contour_image_approx, [approx], -1, (0, 0, 255), 1)
            continue

        contour_area = areas[i]
        # Si el area del contorno no está en el rango valido, se descarta
        if (
            contour_area < (minimum_face_size / 3) ** 2
            or contour_area > maximum_face_size**2
        ):
            cv2.drawContours(contour_image_approx, [approx], -1, (255, 0, 0), 1)
            continue

        cv2.drawContours(contour_image_approx, [approx], -1, (0, 255, 0), 1)
        # Se dibuja el contorno en una nueva imagen
        cv2.drawContours(filtered_contours, [approx], -1, (255,), -1)

    if extra_debug:
        cv2.imshow("contours", contour_image)
        cv2.imshow("filtered contours", filtered_contours)
    if debug:
        cv2.imshow("contours approx", contour_image_approx)

    # Cerrar agujeros entre stickers/cuadrados
    filtered_contours = cv2.morphologyEx(
        filtered_contours, cv2.MORPH_CLOSE, utils.ones_circle(31)
    )
    if extra_debug:
        cv2.imshow("blurred filtered contours", filtered_contours)

    # Buscar contornos para encontrar cara del cubo
    contours, hierarchy = cv2.findContours(
        filtered_contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    areas = []

    for contour in contours:
        # Aproximar contorno a figura simple
        epsilon = 0.08 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # Descartar contornos no rectangulares
        if len(approx) != 4:
            continue
        contour_area = cv2.contourArea(contour)
        areas.append((contour_area, approx))

    if len(areas) == 0:
        return None

    # Elegir el contorno con mayor area
    face_contour = max(areas, key=lambda i: i[0])[1]

    output_roi = img.copy()
    cv2.drawContours(output_roi, [face_contour], -1, (0, 255, 0), 5)
    if debug:
        cv2.imshow("output", output_roi)

    return Quad(*(Point(*face_contour[i, 0, :]) for i in range(4)))
