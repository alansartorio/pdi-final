from typing import Iterable, NamedTuple

from cv2.typing import MatLike
from numpy.typing import NDArray
import numpy as np
import cv2
import math


Image = NDArray[np.uint8]


def triple[T](iterable: Iterable[T]) -> tuple[T, T, T]:
    res = tuple(iterable)
    assert len(res) == 3
    return res


Point = NamedTuple("Point", [("x", int), ("y", int)])
Quad = NamedTuple(
    "Quad",
    [
        ("tl", Point),
        ("tr", Point),
        ("br", Point),
        ("bl", Point),
    ],
)


def rectangle(start: Point, end: Point) -> Quad:
    return Quad(
        Point(start.x, start.y),
        Point(end.x, start.y),
        Point(end.x, end.y),
        Point(start.x, end.y),
    )


def crop_quad(img: Image, quad: Quad, width: int, height: int) -> Image:
    pts1 = np.array(
        (quad.tl, quad.tr, quad.br, quad.bl),
        dtype=np.float32,
    )
    pts2 = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )

    M = cv2.getPerspectiveTransform(pts1, pts2)

    return cv2.warpPerspective(img, M, (width, height))  # type: ignore


def scale_matrix(rx: float, ry: float):
    return np.array(
        [
            [rx, 0, 0],
            [0, ry, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )


def translation_matrix(tx: float, ty: float):
    return np.array(
        [
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )


def normalize_vec2(vec: NDArray[np.float32]):
    return vec / vec[2]


def crop_square(img, size, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    min_size = np.amin([h, w])

    # Centralize and crop
    crop_img = img[
        int(h / 2 - min_size / 2) : int(h / 2 + min_size / 2),
        int(w / 2 - min_size / 2) : int(w / 2 + min_size / 2),
    ]
    resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)

    return resized


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
