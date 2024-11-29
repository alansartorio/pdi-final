from typing import Any
import numpy as np
import cv2
import glob

from find_face import find_face
from plot import FaceProperties, extract_face, get_stickers, overlay_face
from utils import Image


# img: Image = cv2.imread("preprocessed/square_cold_noflash_02.jpg")  # type: ignore
images: dict[str, Image] = {file: cv2.imread(file) for file in glob.glob("preprocessed/*.jpg")} # type: ignore
# images: dict[str, Image] = {file: cv2.imread(file) for file in glob.glob("preprocessed/square_cold_flash_08.jpg")} # type: ignore

for file, img in images.items():
    quad = find_face(img)

    if quad is None:
        print(f"could not find face in image {file}")
        continue

    height, width = img.shape[:2]
    assert height == width
    img_size = width
    face_size = int(width * 0.4)

    face_properties = FaceProperties(quad, 0.9)
    face = extract_face(get_stickers(img, face_properties))

    overlayed = overlay_face(
        img,
        face_properties,
        face,
    )
    cv2.imshow(f"overlay_{file}", overlayed)

while cv2.waitKey(0) != ord("q"):
    pass

exit(0)

