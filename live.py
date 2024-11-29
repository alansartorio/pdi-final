import cv2

from find_face import find_face
from plot import FaceProperties, extract_face, get_stickers, overlay_face
from utils import Image

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    img: Image
    ret, img = cam.read()  # type: ignore
    # img = crop_square(img, 300)
    quad = find_face(img, debug=True, extra_debug=True)
    if quad is not None:
        face_properties = FaceProperties(quad, 0.9)
        face = extract_face(get_stickers(img, face_properties))

        overlayed = overlay_face(
            img,
            face_properties,
            face,
        )
        cv2.imshow("overlay", overlayed)

    if cv2.waitKey(10) == ord("q"):
        exit(0)
