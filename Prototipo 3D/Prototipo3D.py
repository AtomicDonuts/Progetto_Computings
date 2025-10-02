# %%
'''
Pasquale Napoli 
'''
from dataclasses import dataclass
from typing import Any
import random as rng
import numpy as np
# import matplotlib.pylab as plt

DIMENSION_RESOLUTION = 1e-4


# %%
@dataclass
class Point3D:
    """
    Classe per i punti in uno spazio tridimensionale.
    Prende in inpit cordinate polari.

       cord_x : float
       cord_y : float
       cord_z : float
    """

    cord_x: float
    cord_y: float
    cord_z: float

    # def set_cord(self, *args):
    #     '''
    #     set_cord _summary_
    #     '''
    #     pass

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

    def __add__(self,point : "Point3D") -> "Point3D":
        return Point3D(
            self.cord_x + point.cord_x,
            self.cord_y + point.cord_y,
            self.cord_z + point.cord_z,
        )


# %%
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
        _vx = np.cos(self.phi) * np.cos(self.theta)
        _vy = np.cos(self.phi) * np.sin(self.theta)
        _vz = np.sin(self.phi)
        return (_vx, _vy, _vz)

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


# %%
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


# %%
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


# %%
class Paralleogram:
    """
    _summary_
    """

    def __init__(self) -> None:
        """
        __init__ _summary_
        """
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_z = 0.0
        self.set_position((0,0,0))
        self.set_dimensions(0,0,0)

    def set_position(self, point: tuple) -> None:
        """
        set_position _summary_

        Args:
            point (tuple): _description_
        """
        self.pos_x = point[0]
        self.pos_y = point[1]
        self.pos_z = point[2]
        self.vertex1 = Point3D(self.pos_x,self.pos_y,self.pos_z)

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
        self.vertex2 = Point3D(
            self.pos_x + self.dir_x,
            self.pos_y + self.dir_y,
            self.pos_z + self.dir_z
        )

    def intersect_with_line(self, line: Line):
        """
        intersect_with_line:
        Slab Method AABB 

        Args:
            line (Line): _description_
        """
        # x_cord_1 = (self.pos_x - line.originpos.cord_x)/line.d_vector[0]
        # x_cord_2 = (self.pos_x + self.dir_x - line.originpos.cord_x) / line.d_vector[0]
        # x_close = np.min((x_cord_1, x_cord_2))
        # x_far = np.max((x_cord_1, x_cord_2))
        cord = self.vertex1 - line.originpos

# %%
class Absorber(Paralleogram):
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
