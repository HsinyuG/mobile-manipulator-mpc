import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import math

class Obstacles:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius