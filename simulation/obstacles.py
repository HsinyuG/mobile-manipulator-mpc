from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle
from mpscenes.obstacles.cylinder_obstacle import CylinderObstacle

import os

movable_sphere1_dict = {
    "type": "sphere",
    "geometry": {"position": [1.0, -2.0, 2.0], "radius": 0.2},
    "rgba": [0.3, 0.5, 0.6, 1.0],
    'movable': True,
}
movable_sphere1 = SphereObstacle(name="simpleSphere", content_dict=movable_sphere1_dict)

static_cylinder_dict = {
    "type": "cylinder",
    "movable": False,
    "geometry": {
        "position": [-1.0, -3.0, 0.0],
        "radius": 0.3,
        "height": 2.0,
    },
    "rgba": [0.1, 0.3, 0.3, 1.0],
}
static_cylinder = CylinderObstacle(name="cylinder_obstacle", content_dict=static_cylinder_dict)

static_box1_dict = {
    'type': 'box',
    'geometry': {
        'position' : [0.5, -2.0, 0.5],
        'orientation': [0.923, 0, 0, -0.38],
        'width': 0.5,
        'height': 0.5,
        'length': 0.5,
    },
    'movable': False,
}
static_box1 = BoxObstacle(name="movable_box", content_dict=static_box1_dict)

movable_box1_dict = {
    'type': 'box',
    'geometry': {
        'position' : [0.5, -0.5, 1.5],
        'orientation': [0.923, 0, 0, -0.38],
        'width': 0.5,
        'height': 0.5,
        'length': 0.5,
    },
    'movable': True,
}
movable_box1 = BoxObstacle(name="movable_box", content_dict=movable_box1_dict)
