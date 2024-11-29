
from typing import Iterable, NamedTuple

from numpy.typing import NDArray
import numpy as np


Image = NDArray[np.uint8]

def triple[T](iterable: Iterable[T]) -> tuple[T, T, T]:
    res = tuple(iterable)
    assert len(res) == 3
    return res

Point = NamedTuple("Point", [("x", int), ("y", int)])
Rectangle = NamedTuple("Rectangle", [("start", Point), ("end", Point)])

def crop_rectangle(img: Image, rect: Rectangle) -> Image:
    return img[rect.start.y : rect.end.y, rect.start.x : rect.end.x, :]

