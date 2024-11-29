from typing import Optional
import cv2
import numpy as np
from cv2.typing import MatLike
import math

from utils import Point, Quad


def find_closest(value_list: list[float], n: int, expand_with_delta: float = 0):
    l = sorted(enumerate(value_list), key=lambda i: i[1])
    window_dists = {
        tuple(map(lambda item: item[0], l[i : i + n])): l[i + n - 1][1] - l[i][1]
        for i in range(len(l) - n + 1)
    }

    window_index = min(enumerate(window_dists.items()), key=lambda i: i[1][1])[0]
    first = window_index
    window_first = l[first]
    last = window_index + n - 1
    window_last = l[last]

    # expand selection using delta
    while first > 0 and l[first - 1][1] > window_first[1] - expand_with_delta:
        first -= 1
    while last < len(l) - 1 and window_last[1] + expand_with_delta > l[last + 1][1]:
        last += 1

    return tuple(original_index for original_index, _ in l[first : last + 1])


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

def find_face(img: MatLike, debug=False, extra_debug=False) -> Optional[Quad]:
    height, width = img.shape[:2]
    img_size = min(width, height)
    minimum_face_size = 0.15 * img_size
    maximum_face_size = 0.98 * img_size

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if extra_debug:
        cv2.imshow("original", img)

    # _ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(img, 100, 130)
    # edges = cv2.blur(edges, (9, 9))
    ret, edges = cv2.threshold(255 - edges, 240, 255, cv2.THRESH_BINARY)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((12, 12), dtype=np.uint8))
    edges = cv2.erode(edges, np.ones((3, 3), dtype=np.uint8))

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
        if children_area > 0 and children_area > minimum_face_size ** 2:
            # print(children_area)
            cv2.drawContours(contour_image_approx, [contour], -1, (255, 0, 255), 1)
            cv2.drawContours(contour_image_approx, contours, first_child, (0, 255, 255), 3)
            continue
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        has_non_right_angle = True
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

    # cv2.ContourApproximationModes

    if extra_debug:
        cv2.imshow("contours", contour_image)
        cv2.imshow("filtered contours", filtered_contours)
    if debug:
        cv2.imshow("contours approx", contour_image_approx)

    filtered_contours = cv2.morphologyEx(filtered_contours, cv2.MORPH_CLOSE, np.ones((31, 31), dtype=np.uint8))
    # filtered_contours = cv2.blur(filtered_contours, (21, 21))
    # ret, filtered_contours = cv2.threshold(filtered_contours, 100, 255, cv2.THRESH_BINARY)
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
