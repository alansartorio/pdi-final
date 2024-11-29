from typing import Any
import cv2
import numpy as np
from plot import FaceProperties, extract_face, get_stickers, overlay_face
from utils import Point, Quad


img: np.ndarray[Any, np.dtype[np.uint8]] = cv2.imread("preprocessed/transformed.jpg")  # type: ignore

height, width = img.shape[:2]
assert height == width
img_size = width
face_size = int(width * 0.4)

# face_properties = FaceProperties.centered(width, height, face_size, 0.9)
face_properties = FaceProperties(
    Quad(
        Point(182, 205),
        Point(263, 205),
        Point(296, 228),
        Point(236, 242),
    ),
    0.9,
)

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
