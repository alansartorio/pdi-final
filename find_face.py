from typing import Optional
import cv2
import numpy as np
from cv2.typing import MatLike
import math

from utils import Point, Quad


def all_right_angles(shape: MatLike):
    assert len(shape) >= 3
    cycled_approx = np.append(shape.copy(), [shape[0], shape[1]], axis=0)
    for a, b, c in zip(cycled_approx, cycled_approx[1:], cycled_approx[2:]):
        ab = b[0] - a[0]
        bc = c[0] - b[0]
        inner = np.dot(ab, bc) / (np.linalg.vector_norm(ab) * np.linalg.vector_norm(bc))
        if inner < -1 or inner > 1:
            return False
        angle = math.acos(inner)
        right_angle = abs(angle - math.pi / 2) < math.pi / 10
        almost_360 = abs(angle - math.pi * 2) < math.pi / 5
        almost_180 = abs(angle - math.pi) < math.pi / 5
        if not right_angle and not almost_360 and not almost_180:
            return False
    return True


def ones_circle(size: int):
    radius = size // 2
    arr = np.arange(-radius, radius + 1) ** 2
    out = np.add.outer(arr, arr) < radius**2
    return out.astype(np.uint8)


def does_contour_touch_border(contour: MatLike, width, height):
    for ((x, y),) in contour:
        if x == 0 or y == 0 or x == width - 1 or y == height - 1:
            return True

    return False


def find_face(img: MatLike, debug=False, extra_debug=False) -> Optional[Quad]:
    height, width = img.shape[:2]
    img_size = min(width, height)
    minimum_face_size = 0.10 * img_size
    maximum_face_size = 0.98 * img_size

    if extra_debug:
        cv2.imshow("original", img)

    edges = cv2.Canny(img, 100, 170)
    ret, edges = cv2.threshold(255 - edges, 240, 255, cv2.THRESH_BINARY)

    if extra_debug:
        cv2.imshow("edges1", edges)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, ones_circle(4))

    if extra_debug:
        cv2.imshow("edges2", edges)
    edges = cv2.erode(edges, ones_circle(4))

    if debug:
        cv2.imshow("edges", edges)

    contour_image = np.zeros_like(img)

    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    hierarchy = tuple(*hierarchy)

    contour_image_approx = np.zeros_like(img)

    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)
    areas = [cv2.contourArea(contour) for contour in contours]

    filtered_contours = np.zeros_like(edges)

    for (i, contour), (next, previous, first_child, parent) in zip(
        enumerate(contours), hierarchy
    ):
        children_area = 0
        next = first_child
        while next != -1:
            children_area += areas[next]
            next = hierarchy[next][0]
        if children_area > 0 and children_area > minimum_face_size**2:
            cv2.drawContours(contour_image_approx, [contour], -1, (255, 0, 255), 1)
            cv2.drawContours(
                contour_image_approx, contours, first_child, (0, 255, 255), 3
            )
            continue
        if does_contour_touch_border(contour, width, height):
            continue
        # epsilon = minimum_face_size / 3 * 4 * .1
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 4:
            cv2.drawContours(contour_image_approx, [approx], -1, (0, 0, 255), 1)
            continue

        if not all_right_angles(approx):
            cv2.drawContours(contour_image_approx, [approx], -1, (0, 0, 255), 1)
            continue
        contour_area = areas[i]
        if (
            contour_area < (minimum_face_size / 3) ** 2
            or contour_area > maximum_face_size**2
        ):
            cv2.drawContours(contour_image_approx, [approx], -1, (255, 0, 0), 1)
            continue
        cv2.drawContours(contour_image_approx, [approx], -1, (0, 255, 0), 1)
        cv2.drawContours(filtered_contours, [approx], -1, (255,), -1)

    if extra_debug:
        cv2.imshow("contours", contour_image)
        cv2.imshow("filtered contours", filtered_contours)
    if debug:
        cv2.imshow("contours approx", contour_image_approx)

    filtered_contours = cv2.morphologyEx(
        filtered_contours, cv2.MORPH_CLOSE, np.ones((31, 31), dtype=np.uint8)
    )
    if extra_debug:
        cv2.imshow("blurred filtered contours", filtered_contours)

    contours, hierarchy = cv2.findContours(
        filtered_contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    areas = []

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        epsilon = 0.08 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) != 4:
            continue
        areas.append((contour_area, approx))

    if len(areas) == 0:
        return None
    face_contour = max(areas, key=lambda i: i[0])[1]

    output_roi = img.copy()
    cv2.drawContours(output_roi, [face_contour], -1, (0, 255, 0), 5)
    if debug:
        cv2.imshow("output", output_roi)

    return Quad(*(Point(*face_contour[i, 0, :]) for i in range(4)))
