# %%
from typing import Any
import random as rng
import numpy as np
import matplotlib.pylab as plt
from dataclasses import dataclass

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

    def set_cord(self,*args):
        pass

    def magnitude(self):
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
        other_x, other_y, other_z = None,None,None

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
                "Invalid Input. Use Point3D, 3 element " \
                "tuple/list or 3 float numbers"
             )
        
    def __getitem__(self,
                    index: int) -> float:
        _points = (self.cord_x,self.cord_y,self.cord_z)
        return _points[index]

# %%
class Line:
    def __init__(self) -> None:
        self.originpos = Point3D(0, 0, 0)
        self.theta = 0.
        self.phi = 0.
        self.generate()

    def set_origin(self,
                   coord):
        if len(coord) == 3:
            self.originpos = coord
        else:
            raise TypeError("coord must have 3 element ")

    def generate(self) -> None:
        self.originpos = Point3D(rng.random(), rng.random(), rng.random())
        self.theta = rng.uniform(0.0, np.pi)
        self.phi = rng.uniform(0.0, 2 * np.pi)
        self._direction_vector()

    def _direction_vector(self):
        _vx = np.cos(self.phi) * np.cos(self.theta)
        _vy = np.cos(self.phi) * np.sin(self.theta)
        _vz = np.sin(self.phi)
        self.d_vector = (_vx,_vy,_vz)

    def is_point_on_line(self,
                         point: Point3D) -> bool:
        _var1 = (self.originpos.cord_x - point.cord_x) / self.d_vector[0]
        _var2 = (self.originpos.cord_y - point.cord_y) / self.d_vector[1]
        _var3 = (self.originpos.cord_z - point.cord_z) / self.d_vector[2]
        if _var1 == _var2 and _var1 == _var3:
            return True
        return False

# %%
class Particle(Line):

    def __init__(self) -> None:
        super().__init__()
        self.energy = 0.
        self.mass = 0.
        self.charge = 0
        self.momentum = 0. 
        self.decayed = False
        # da finire

    def decay(self, time_passed: float) -> None:
        if self.decayed:
            return None
        # vedere se decade in qualche modo

# %%
class Muon(Particle):
    def __init__(self) -> None:
        super().__init__()
        self.mass = 105.66 #MeV
        self.charge = -1

# %%
class Paralleogram:

    def __init__(self) -> None:
        self.pos_x = 0.
        self.pos_y = 0.
        self.pos_z = 0.
        self.dir_x = 0.
        self.dir_y = 0.
        self.dir_z = 0.

    def set_position(self,
                     point: tuple) -> None:
        self.pos_x = point[0]
        self.pos_y = point[1]
        self.pos_z = point[2]

    def set_dimensions(self,
                       dir_x : float,
                       dir_y : float,
                       dir_z : float) -> None:
        self.dir_x = dir_x
        self.dir_y = dir_y
        self.dir_z = dir_z

    def intersect_with_line(self,
                            line : Line):
        inside = False
        start = Point3D(0,0,0)
        end = Point3D(0,0,0)
        for depth in np.arange(self.pos_z,self.pos_z + self.dir_z,DIMENSION_RESOLUTION):
            z_var = (depth - line.originpos[2]) / line.d_vector[2]
            x_var = line.d_vector[0] * z_var + line.originpos[0]
            y_var = line.d_vector[1] * z_var + line.originpos[1]
            if x_var > self.pos_x and x_var < self.pos_x + self.dir_x:
                if y_var > self.pos_y and y_var < self.pos_y + self.dir_y:
                    if not inside:
                        inside = True
                        start = Point3D(x_var,y_var,z_var)
                    
                        

        

# %%
class Absorber(Paralleogram):

    def __init__(self,
                 density : float,
                 atomic_number : int,
                 atomic_mass : int) -> None:
        self.density = density
        self.atomic_number = atomic_number
        self.atomic_mass = atomic_mass

    def max_traveling_distance(self,
                               particle: Muon) -> float:
        # calcolare la distanza massima
        distance = 0
        return distance 
    
    def assorbing(self,
                  particle: Muon) -> None:
        
        rng.uniform(0.,self.max_traveling_distance(particle))            

# %%
pino = Absorber(0.5,1,1)

# %%
pino.set_dimensions(4,4,4)

# %%
pino.set_position((0,0,0))

# %%
Muon


