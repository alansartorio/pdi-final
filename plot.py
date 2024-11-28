from collections.abc import Iterable
from enum import Enum
from typing import Any, NamedTuple
import cv2
import numpy as np
from dataclasses import dataclass

from numpy._typing import NDArray


class Color(Enum):
    RED = 0
    BLUE = 1
    GREEN = 2
    ORANGE = 3
    YELLOW = 4
    WHITE = 5

    def get_rgb(self) -> tuple[int, int, int]:
        match self:
            case Color.RED:
                return (255, 0, 0)
            case Color.BLUE:
                return (0, 0, 255)
            case Color.GREEN:
                return (0, 255, 0)
            case Color.ORANGE:
                return (255, 100, 0)
            case Color.YELLOW:
                return (255, 255, 0)
            case Color.WHITE:
                return (255, 255, 255)

    def get_bgr(self) -> tuple[int, int, int]:
        reversed_color = tuple(reversed(self.get_rgb()))
        assert len(reversed_color) == 3
        return reversed_color


Row = tuple[Color, Color, Color]
Face = tuple[Row, Row, Row]


def char_to_color(char: str) -> Color:
    match char:
        case "r":
            return Color.RED
        case "b":
            return Color.BLUE
        case "g":
            return Color.GREEN
        case "o":
            return Color.ORANGE
        case "y":
            return Color.YELLOW
        case "w":
            return Color.WHITE
    assert False


def parse_row(row_str: str) -> Row:
    parsed = tuple(map(char_to_color, row_str))
    assert len(parsed) == 3
    return parsed


def parse_face(face_str: str) -> Face:
    parsed = tuple(map(parse_row, face_str.split("\n")))
    assert len(parsed) == 3
    return parsed


Point = NamedTuple("Point", [("x", int), ("y", int)])
Rectangle = NamedTuple("Rectangle", [("start", Point), ("end", Point)])


@dataclass
class FaceProperties:
    rectangle: Rectangle
    sticker_margin: int

    @classmethod
    def centered(cls, width: int, height: int, face_size: int, sticker_margin: int):
        cy, cx = (height // 2, width // 2)
        half_face = face_size // 2
        return cls(
            Rectangle(
                Point(cx - half_face, cy - half_face),
                Point(cx + half_face, cy + half_face),
            ),
            sticker_margin,
        )

    def get_sticker_rectangle(self, x: int, y: int) -> Rectangle:
        dir = Point(
            self.rectangle.end.x - self.rectangle.start.x,
            self.rectangle.end.y - self.rectangle.start.y,
        )
        tile_dir = Point(dir.x // 3, dir.y // 3)
        tile_start = Point(
            self.rectangle.start.x + tile_dir.x * x + self.sticker_margin,
            self.rectangle.start.y + tile_dir.y * y + self.sticker_margin,
        )
        tile_end = Point(
            self.rectangle.start.x + tile_dir.x * (x + 1) - self.sticker_margin,
            self.rectangle.start.y + tile_dir.y * (y + 1) - self.sticker_margin,
        )

        return Rectangle(tile_start, tile_end)


def overlay_face(img, face_properties: FaceProperties, face: Face):
    for y, row in enumerate(face):
        for x, tile in enumerate(row):
            tile_start, tile_end = face_properties.get_sticker_rectangle(x, y)
            cv2.rectangle(
                img,
                tile_start,
                tile_end,
                tile.get_bgr(),
                4,
            )
    return img


Image = NDArray[np.uint8]
StickerImage = Image
RowImages = tuple[StickerImage, StickerImage, StickerImage]
FaceImages = tuple[RowImages, RowImages, RowImages]


def crop_rectangle(img: Image, rect: Rectangle) -> StickerImage:
    return img[rect.start.y : rect.end.y, rect.start.x : rect.end.x, :]


def triple[T](iterable: Iterable[T]) -> tuple[T, T, T]:
    res = tuple(iterable)
    assert len(res) == 3
    return res


def get_stickers(img: Image, face_properties: FaceProperties) -> FaceImages:
    return triple(
        triple(
            crop_rectangle(img, face_properties.get_sticker_rectangle(x, y))
            for x in range(3)
        )
        for y in range(3)
    )



img: np.ndarray[Any, np.dtype[np.uint8]] = cv2.imread("preprocessed/square_cold_flash_08.jpg") # type: ignore

height, width = img.shape[:2]
assert height == width
img_size = width
face_size = int(width * 0.4)

face_properties = FaceProperties.centered(width, height, face_size, 4)

for y, row in enumerate(get_stickers(img, face_properties)):
    for x, sticker in enumerate(row):
        cv2.imshow(f"sticker_{x},{y}", sticker)

overlayed = overlay_face(
    img,
    face_properties,
    parse_face("ggr\ngyy\nrwg"),
)
cv2.imshow("overlay", overlayed)
while cv2.waitKey() != 27:
    pass
