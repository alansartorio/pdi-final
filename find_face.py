from typing import Optional
import cv2
import numpy as np
from cv2.typing import MatLike
import math

from utils import Point, Quad


def find_closest(value_list: list[float], n: int, expand_with_delta: float = 0):
    l = sorted(enumerate(value_list), key=lambda i: i[1])
    window_dists = {
        tuple(map(lambda item: item[0], l[i : i + n])): l[i + n - 1][1] - l[i][1]
        for i in range(len(l) - n + 1)
    }

    window_index = min(enumerate(window_dists.items()), key=lambda i: i[1][1])[0]
    first = window_index
    window_first = l[first]
    last = window_index + n - 1
    window_last = l[last]

    # expand selection using delta
    while first > 0 and l[first - 1][1] > window_first[1] - expand_with_delta:
        first -= 1
    while last < len(l) - 1 and window_last[1] + expand_with_delta > l[last + 1][1]:
        last += 1

    return tuple(original_index for original_index, _ in l[first : last + 1])


def find_face(img: MatLike) -> Optional[Quad]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cv2.imshow("original", img)
    cv2.imshow("hsv", hsv)

    height, width = img.shape[:2]
    scale_factor = height / 300
    vivid_colors = np.zeros((height, width, 1), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            if (
                hsv[y, x, 1] > 100 and hsv[y, x, 2] > 100
                # or (hsv[y, x, 1] < 20 and hsv[y, x, 2] > 165)
            ):
                vivid_colors[y, x, :] = 255
    cv2.imshow("vivid_colors", vivid_colors)

    face_blob = vivid_colors.copy()
    blur_size = int(20 * scale_factor) * 2 + 1
    face_blob = cv2.blur(face_blob, (blur_size, blur_size))
    # cv2.imshow("blur", face_blob)
    _ret, face_blob = cv2.threshold(face_blob, 50, 255, cv2.THRESH_BINARY)
    # print(face_blob)
    # cv2.imshow("thresh", face_blob)

    with_contours = img.copy()
    contours, hierarchy = cv2.findContours(
        face_blob, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        return None
    cv2.drawContours(with_contours, contours, -1, (0, 255, 0), 3)
    # cv2.imshow("contours", with_contours)

    max_contour = max(contours, key=lambda contour: cv2.contourArea(contour))
    bbx, bby, bbwidth, bbheight = cv2.boundingRect(max_contour)
    margin = int(20 * scale_factor)
    bbx -= margin
    bby -= margin
    bbwidth += margin * 2
    bbheight += margin * 2
    if bbx < 0:
        bbwidth += abs(bbx)
        bbx = 0
    if bby < 0:
        bbheight += abs(bby)
        bby = 0
    # print(bbx, bby, bbwidth, bbheight)
    if bbwidth == 0 or bbheight == 0:
        return None

    roi = vivid_colors[bby : bby + bbheight, bbx : bbx + bbwidth, :]

    cv2.imshow("roi", roi)

    roi_scale_factor = 180 / min(bbwidth, bbheight)

    # Edge detection
    edges = cv2.Canny(roi, 20, 200, None, 3)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel)

    cv2.imshow("edges", edges)

    cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cdst2 = cdst.copy()

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        int(40 * roi_scale_factor),
        None,
        int(20 * roi_scale_factor),
        int(40 * roi_scale_factor),
    )

    line_lengths = []
    if lines is None:
        return None
    for i in range(0, len(lines)):
        l = lines[i][0]
        line_lengths.append(math.sqrt((l[0] - l[2]) ** 2 + (l[1] - l[3]) ** 2))
        cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow("lines", cdst)

    if len(line_lengths) < 4:
        return None
    line_indices = find_closest(line_lengths, 4, expand_with_delta=2 * roi_scale_factor)
    print(line_lengths[line_indices[len(line_indices) // 2]], line_indices)

    for i in line_indices:
        l = lines[i][0]
        cv2.line(cdst2, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow("lines2", cdst2)

    return Quad(
        Point(0, 0),
        Point(100, 0),
        Point(100, 100),
        Point(0, 100),
    )
