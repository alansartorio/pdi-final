from typing import Any
import cv2
import numpy as np
from numpy.typing import NDArray

from color import Color
from utils import (
    Image,
    Point,
    Quad,
    crop_quad,
    normalize_vec2,
    rectangle,
    scale_matrix,
    triple,
)

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


class FaceProperties:
    quad: Quad
    sticker_size: float  # 0 to 1
    perspective_transform: NDArray[np.float32]
    inverted_perspective_transform: NDArray[np.float32]

    def __init__(self, quad: Quad, sticker_size: float) -> None:
        self.quad = quad
        self.sticker_size = sticker_size
        pts1 = np.array(quad, dtype=np.float32)
        pts2 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        self.perspective_transform = cv2.getPerspectiveTransform(pts1, pts2)  # type: ignore
        self.inverted_perspective_transform = np.linalg.inv(self.perspective_transform)

    @classmethod
    def centered(cls, width: int, height: int, face_size: int, sticker_size: float):
        cy, cx = (height // 2, width // 2)
        half_face = face_size // 2
        return cls(
            rectangle(
                Point(cx - half_face, cy - half_face),
                Point(cx + half_face, cy + half_face),
            ),
            sticker_size,
        )

    def get_sticker_quad(self, x: int, y: int) -> Quad:
        return Quad(
            *(
                Point(
                    *normalize_vec2(
                        self.inverted_perspective_transform
                        @ scale_matrix(1 / 3, 1 / 3)
                        @ np.append(
                            (np.array(point, dtype=np.float32) - 0.5)
                            * self.sticker_size
                            + 0.5
                            + np.array((x, y), np.float32),
                            [1],
                        )
                    )[:2]
                )
                for point in ((0, 0), (1, 0), (1, 1), (0, 1))
            )
        )


def overlay_face(img, face_properties: FaceProperties, face: Face):
    for y, row in enumerate(face):
        for x, tile in enumerate(row):
            quad = np.array(face_properties.get_sticker_quad(x, y), dtype=np.int32)
            cv2.drawContours(
                img,
                [quad],
                0,
                tile.get_bgr(),
                2,
            )
    return img


StickerImage = Image
RowImages = tuple[StickerImage, StickerImage, StickerImage]
FaceImages = tuple[RowImages, RowImages, RowImages]


def get_stickers(img: Image, face_properties: FaceProperties) -> FaceImages:
    return triple(
        triple(
            crop_quad(img, face_properties.get_sticker_quad(x, y), 20, 20)
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
