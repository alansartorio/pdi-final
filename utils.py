from typing import Iterable, NamedTuple

from numpy.typing import NDArray
import numpy as np
import cv2


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
