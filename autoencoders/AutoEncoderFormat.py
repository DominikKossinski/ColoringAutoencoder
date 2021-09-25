from enum import Enum


class AutoEncoderFormat(Enum):
    RGB = "RGB"
    HSV = "HSV"
    LAB = "LAB"

    def __str__(self):
        return self.value
