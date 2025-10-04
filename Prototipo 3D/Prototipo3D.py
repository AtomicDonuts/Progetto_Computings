"""
Pasquale Napoli
"""

from typing import Any
import random as rng
from loguru import logger
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# import matplotlib.pylab as plt

DIMENSION_RESOLUTION = 1e-4


class Point3D:
    """
    Classe per i punti in uno spazio tridimensionale.
    Prende in inpit cordinate polari.

       cord_x : float
       cord_y : float
       cord_z : float
    """

    def __init__(self, cord_x, cord_y, cord_z) -> None:
        self.cord_x = cord_x
        self.cord_y = cord_y
        self.cord_z = cord_z
        self.np_cord = np.array([cord_x, cord_y, cord_z], dtype=float)

    def magnitude(self) -> Any:
        """
        magnitude:
        Distanza del punto dall'origine

        Returns:
            _type_: _description_
        """
        return np.linalg.norm((self.cord_x, self.cord_y, self.cord_z))

    def polar(self) -> tuple:
        """
        Restituisce le coordinate polari
        del punto in una tupla del tipo (r,theta,phi)
        """
        radius = self.magnitude()
        theta = np.arccos(self.cord_z / radius)
        phi = np.arctan2(self.cord_y, self.cord_x)
        return (radius, theta, phi)

    def distance_from(self, *args):
        """
        Calcola la distanza da un altro punto, che può essere fornito
        in diversi formati:
        1. Un altro oggetto Point3D
        2. Una tupla o lista di 3 coordinate (es. (1, 2, 3))
        3. Tre argomenti numerici separati (es. 1, 2, 3)

        Returns:
            float: distance from the two points
        """
        other_x, other_y, other_z = None, None, None

        if len(args) == 1:
            other = args[0]
            if isinstance(other, Point3D):
                other_x, other_y, other_z = other.cord_x, other.cord_y, other.cord_z

            elif isinstance(other, (tuple, list)) and len(other) == 3:
                other_x, other_y, other_z = other

        elif len(args) == 3:
            other_x, other_y, other_z = args

        if other_x is not None and other_y is not None and other_z is not None:
            return np.linalg.norm(
                (
                    self.cord_x - float(other_x),
                    self.cord_y - float(other_y),
                    self.cord_z - float(other_z),
                )
            )
        else:
            raise TypeError(
                "Invalid Input. Use Point3D, 3 element tuple/list or 3 float numbers"
            )

    def __getitem__(self, index: int) -> float:
        _points = (self.cord_x, self.cord_y, self.cord_z)
        return _points[index]

    def __str__(self) -> str:
        return f"Point3D: x:{self.cord_x} y:{self.cord_y} z:{self.cord_z}"


class Line:
    """
    _summary_
    """

    def __init__(self) -> None:
        self.originpos = Point3D(0, 0, 0)
        self.theta = 0.0
        self.phi = 0.0
        self.generate()
        self.d_vector = self._direction_vector()

    def set_origin(self, coord):
        """
        set_origin _summary_

        Args:
            coord (_type_): _description_

        Raises:
            TypeError: _description_
        """
        if len(coord) == 3:
            self.originpos = coord
        else:
            raise TypeError("coord must have 3 element ")

    def generate(self) -> None:
        """
        generate _summary_
        """
        self.originpos = Point3D(rng.random(), rng.random(), rng.random())
        self.theta = rng.uniform(0.0, np.pi)
        self.phi = rng.uniform(0.0, 2 * np.pi)
        self._direction_vector()

    def _direction_vector(self):
        """
        _direction_vector _summary_

        Returns:
            _type_: _description_
        """
        _vx = np.sin(self.theta) * np.cos(self.phi)
        _vy = np.sin(self.theta) * np.sin(self.phi)
        _vz = np.cos(self.theta)
        return np.array((_vx, _vy, _vz), dtype=float)

    def is_point_on_line(self, point: Point3D) -> bool:
        """
        is_point_on_line _summary_

        Args:
            point (Point3D): _description_

        Returns:
            bool: _description_
        """
        _var1 = (self.originpos.cord_x - point.cord_x) / self.d_vector[0]
        _var2 = (self.originpos.cord_y - point.cord_y) / self.d_vector[1]
        _var3 = (self.originpos.cord_z - point.cord_z) / self.d_vector[2]
        if _var1 == _var2 and _var1 == _var3:
            return True
        return False

    def __call__(self, parameter: float) -> Point3D:
        """
        Restituisce un punto p in cui viene risolta
        l'equazione parametrica vettoriale:
        point = d_vector * parameter + originpos

        Args:
            parameter (float): parametro

        Returns:
            Point3D: _description_
        """
        _point = self.d_vector * parameter + self.originpos.np_cord
        return Point3D(_point[0], _point[1], _point[2])

    def obj3d(self,lin_array: np.ndarray):
        '''
        obj3d _summary_

        Returns:
            _type_: _description_
        '''
        arr_x = self.d_vector[0] * lin_array + self.originpos.cord_x
        arr_y = self.d_vector[1] * lin_array + self.originpos.cord_y
        arr_z = self.d_vector[2] * lin_array + self.originpos.cord_z
        return np.array([arr_x,arr_y,arr_z])

class Particle(Line):
    """
    Particle _summary_

    Args:
        Line (_type_): _description_
    """

    def __init__(self) -> None:
        """
        __init__ _summary_
        """
        super().__init__()
        self.energy = 0.0
        self.mass = 0.0
        self.charge = 0
        self.momentum = 0.0
        self.decayed = False
        # da finire

    def decay(self, time_passed: float):
        """
        decay _summary_

        Args:
            time_passed (float): _description_

        Returns:
            _type_: _description_
        """
        if self.decayed:
            return time_passed
        # vedere se decade in qualche modo


class Muon(Particle):
    """
    Muon _summary_

    Args:
        Particle (_type_): _description_
    """

    def __init__(self) -> None:
        """
        __init__ _summary_
        """
        super().__init__()
        self.mass = 105.66  # MeV
        self.charge = -1


class Parallelepiped:
    """
    Solo parallelepipedi rettangoli (per ora)
    """

    def __init__(self) -> None:
        """
        __init__ _summary_
        """
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_z = 0.0
        self.set_position((0, 0, 0))
        self.set_dimensions(0, 0, 0)
        self.construct_vertex()

    def set_position(self, point: tuple) -> None:
        """
        set_position _summary_

        Args:
            point (tuple): _description_
        """
        self.pos_x = point[0]
        self.pos_y = point[1]
        self.pos_z = point[2]
        self.vertex0 = Point3D(self.pos_x, self.pos_y, self.pos_z)

    def set_dimensions(self, dir_x: float, dir_y: float, dir_z: float) -> None:
        """
        set_dimensions _summary_

        Args:
            dir_x (float): _description_
            dir_y (float): _description_
            dir_z (float): _description_
        """
        self.dir_x = dir_x
        self.dir_y = dir_y
        self.dir_z = dir_z
        self.construct_vertex()

    def construct_vertex(self):
        """
        construct_vertex _summary_
        """
        self.vertex1 = Point3D(self.pos_x + self.dir_x, self.pos_y, self.pos_z)
        self.vertex2 = Point3D(
            self.pos_x + self.dir_x, self.pos_y + self.dir_y, self.pos_z
        )
        self.vertex3 = Point3D(self.pos_x, self.pos_y + self.dir_y, self.pos_z)
        self.vertex4 = Point3D(self.pos_x, self.pos_y, self.pos_z + self.dir_z)
        self.vertex5 = Point3D(
            self.pos_x + self.dir_x, self.pos_y, self.pos_z + self.dir_z
        )
        self.vertex6 = Point3D(
            self.pos_x + self.dir_x, self.pos_y + self.dir_y, self.pos_z + self.dir_z
        )
        self.vertex7 = Point3D(
            self.pos_x, self.pos_y + self.dir_y, self.pos_z + self.dir_z
        )

    def obj3d(self,
              face_colors="gray",
              line_widths=1,
              edge_colors="k",
              alpha_graph=0.1):
        """
        ritorna l'oggetto 3d in modo che matplotlib lo possa plottare
        utilizzando ax.add_collection3d(xxxx.obj3d())
        per un esempio controllare test_parallelepiped_obj3d
        """
        faces = [
            # Faccia inferiore
            [
                self.vertex0.np_cord,
                self.vertex1.np_cord,
                self.vertex2.np_cord,
                self.vertex3.np_cord,
            ],
            # Faccia superiore
            [
                self.vertex4.np_cord,
                self.vertex5.np_cord,
                self.vertex6.np_cord,
                self.vertex7.np_cord,
            ],
            # Faccia posteriore
            [
                self.vertex2.np_cord,
                self.vertex3.np_cord,
                self.vertex7.np_cord,
                self.vertex6.np_cord,
            ],
            # Faccia anteriore
            [
                self.vertex0.np_cord,
                self.vertex1.np_cord,
                self.vertex5.np_cord,
                self.vertex4.np_cord,
            ],
            # Faccia sinistra
            [
                self.vertex0.np_cord,
                self.vertex3.np_cord,
                self.vertex7.np_cord,
                self.vertex4.np_cord,
            ],
            # Faccia destra
            [
                self.vertex1.np_cord,
                self.vertex2.np_cord,
                self.vertex6.np_cord,
                self.vertex5.np_cord,
            ],
        ]

        poly3d = Poly3DCollection(
            faces,
            facecolors=face_colors,
            linewidths=line_widths,
            edgecolors=edge_colors,
            alpha=alpha_graph,
        )
        return poly3d

    def intersect_with_line(self, line: Line):
        """
        intersect_with_line:
        Slab Method AABB
        Syntax from Wikepedia: https://en.wikipedia.org/wiki/Slab_method

        Args:
            line (Line): _description_
        """
        _d_vector = np.where(line.d_vector == 0.0, 1e-12, line.d_vector)
        cord_low = np.divide(self.vertex0.np_cord - line.originpos.np_cord, _d_vector)
        cord_high = np.divide(self.vertex6.np_cord - line.originpos.np_cord, _d_vector)
        cord_close_far = np.dstack((cord_low, cord_high))
        t_close = cord_close_far.min(axis=2).max()
        t_far = cord_close_far.max(axis=2).min()
        # logger.debug(f"t_close <= t_far : {t_close <= t_far}")
        # logger.debug(f"Intersezione close: {line(t_close)}")
        # logger.debug(f"Intersezione far: {line(t_far)}")
        if t_close <= t_far:
            return [True, [line(t_close), line(t_far)]]
        return [False, [None, None]]


class Absorber(Parallelepiped):
    """
    Absorber _summary_

    Args:
        Paralleogram (_type_): _description_
    """

    def __init__(self, density: float, atomic_number: int, atomic_mass: int) -> None:
        super().__init__()
        self.density = density
        self.atomic_number = atomic_number
        self.atomic_mass = atomic_mass

    def max_traveling_distance(self, particle: Muon) -> float:
        """
        max_traveling_distance
        calcolare la distanza massima
        Args:
            particle (Muon): _description_

        Returns:
            float: _description_
        """
        print(particle.mass)
        distance = 0
        return distance

    def assorbing(self, particle: Muon) -> None:
        """
        assorbing _summary_

        Args:
            particle (Muon): _description_
        """
        rng.uniform(0.0, self.max_traveling_distance(particle))
