
import math
from random import random

from numpy import float64
from simSetup import size


class Vector2:
    def __init__(self, x: float64, y: float64):
        self.x = x
        self.y = y

    @classmethod
    def initRandom_normalized(cls) -> 'Vector2':
        """Initialize and return a normalized vector pointing in a random direction"""
        x = random() - .5
        y = random() - .5
        vec = cls(x, y)
        vec.normalizeSelf()

        return vec

    @classmethod
    def initRandom(cls):
        """Initialize and return a vector pointing to a random location on a size x size board"""
        x = random() * size
        y = random() * size
        vec = cls(x, y)
        return vec

    def length(self):
        """Get length of vector"""
        return math.sqrt(self.x ** 2 + self.y**2)

    def normalizeSelf(self):
        """Normalize vector in place"""
        l = self.length()
        self.x = self.x / l
        self.y = self.y / l

    def add(self, vec2: 'Vector2') -> 'Vector2':
        self.x += vec2.x
        self.y += vec2.y

    def rotate(self, angle: float64):
        x = self.x
        self.x = x * math.cos(angle) - math.sin(angle) * self.y
        self.y = x * math.sin(angle) + math.cos(angle) * self.y
        self.normalizeSelf()

    def __mul__(self, factor: float64) -> 'Vector2':
        return Vector2(self.x * factor, self.y * factor)

    def __add__(self, other: 'Vector2') -> 'Vector2':
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Vector2') -> 'Vector2':
        other = Vector2(-1 * other.x, -1 * other.y)
        return self.__add__(other=other)

    def __str__(self):
        return f"Vector2({self.x}, {self.y})"
