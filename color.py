from enum import Enum

from utils import triple


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
        return triple(reversed(self.get_rgb()))

    @classmethod
    def from_char(cls, char: str):
        match char:
            case "r":
                return cls.RED
            case "b":
                return cls.BLUE
            case "g":
                return cls.GREEN
            case "o":
                return cls.ORANGE
            case "y":
                return cls.YELLOW
            case "w":
                return cls.WHITE
        assert False
