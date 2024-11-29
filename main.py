from typing import Any
import cv2
import numpy as np
from find_face import find_face
from plot import FaceProperties, extract_face, get_stickers, overlay_face
from utils import Point, Quad, crop_square


# img: np.ndarray[Any, np.dtype[np.uint8]] = cv2.imread("preprocessed/transformed.jpg")  # type: ignore

# height, width = img.shape[:2]
# assert height == width
# img_size = width
# face_size = int(width * 0.4)


# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, img = cam.read()
    # img = crop_square(img, 300)
    find_face(img)
    if cv2.waitKey(10) == ord("q"):
        exit(0)

# face_properties = FaceProperties.centered(width, height, face_size, 0.9)
face_properties = FaceProperties(
    find_face(img),
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
