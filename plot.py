from enum import Enum
import cv2


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


def overlay_face(img, face: Face):
    print(face)
    height, width = img.shape[:2]
    assert height == width
    cy, cx = (height // 2, width // 2)
    tile_size = int(height * 0.15)
    tile_size_half = tile_size // 2 - 4
    for y, row in enumerate(face):
        for x, tile in enumerate(row):
            tile_cy, tile_cx = (y - 1) * tile_size + cy, (x - 1) * tile_size + cx
            cv2.rectangle(
                img,
                (tile_cx - tile_size_half, tile_cy - tile_size_half),
                (tile_cx + tile_size_half, tile_cy + tile_size_half),
                tile.get_bgr(),
                4,
            )
    cv2.imshow("overlay", img)
    while cv2.waitKey() != 27:
        pass


img = cv2.imread("preprocessed/square_cold_flash_08.jpg")

overlay_face(
    img,
    parse_face("ggr\ngyy\nrwg"),
)
