from collections.abc import Iterable
from typing import Any, NamedTuple
import cv2
import numpy as np
from dataclasses import dataclass

from numpy._typing import NDArray

from color import Color
from utils import Image, Point, Rectangle, crop_rectangle, triple

Row = tuple[Color, Color, Color]
Face = tuple[Row, Row, Row]


def parse_row(row_str: str) -> Row:
    parsed = tuple(map(Color.from_char, row_str))
    assert len(parsed) == 3
    return parsed


def parse_face(face_str: str) -> Face:
    parsed = tuple(map(parse_row, face_str.split("\n")))
    assert len(parsed) == 3
    return parsed



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


StickerImage = Image
RowImages = tuple[StickerImage, StickerImage, StickerImage]
FaceImages = tuple[RowImages, RowImages, RowImages]


def get_stickers(img: Image, face_properties: FaceProperties) -> FaceImages:
    return triple(
        triple(
            crop_rectangle(img, face_properties.get_sticker_rectangle(x, y))
            for x in range(3)
        )
        for y in range(3)
    )


def bgr_to_hsv(color: tuple[int, int, int]) -> tuple[int, int, int]:
    fake_img = np.array(((color,),), dtype=np.uint8)

    return triple(cv2.cvtColor(fake_img, cv2.COLOR_BGR2HSV)[0, 0, :])


def color_diff(color_a: tuple[int, int, int], color_b: tuple[int, int, int]) -> int:
    diff = np.abs(
        np.array(bgr_to_hsv(color_a), dtype=np.int16)
        - np.array(bgr_to_hsv(color_b), dtype=np.int16)
    )
    diff[0] = diff[0] if diff[0] < 90 else 180 - diff[0]
    return np.sum(diff)


def get_sticker_color(img: StickerImage) -> Color:
    avg_color: NDArray[np.uint8] = np.mean(img, axis=(0, 1))
    colors = (
        Color.RED,
        Color.BLUE,
        Color.GREEN,
        Color.ORANGE,
        Color.YELLOW,
        Color.WHITE,
    )
    diffs: dict[Color, int] = dict(
        map(
            lambda color: (color, color_diff(color.get_bgr(), tuple(avg_color))),
            colors,
        )
    )
    return min(diffs.items(), key=lambda v: v[1])[0]


def extract_face(img: FaceImages) -> Face:
    return triple(triple(get_sticker_color(sticker) for sticker in row) for row in img)


img: np.ndarray[Any, np.dtype[np.uint8]] = cv2.imread(
    "preprocessed/square_cold_flash_08.jpg"
)  # type: ignore

height, width = img.shape[:2]
assert height == width
img_size = width
face_size = int(width * 0.4)

face_properties = FaceProperties.centered(width, height, face_size, 4)

face = extract_face(get_stickers(img, face_properties))
print(face)

overlayed = overlay_face(
    img,
    face_properties,
    face,
)
cv2.imshow("overlay", overlayed)
while cv2.waitKey() != 27:
    pass
