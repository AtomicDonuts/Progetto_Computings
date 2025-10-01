# %%
import random as rng
import numpy as np
from typing import Any
from dataclasses import dataclass

# %%
@dataclass
class Point3D:
    '''
    Classe per i punti in uno spazio tridimensionale.
    Prende in inpit cordinate polari.

       cord_x : float
       cord_y : float
       cord_z : float
    '''

    cord_x: float
    cord_y: float
    cord_z: float

    def magnitude(self):
        '''
        Distanza del punto dall'origine

        Returns:
            float 
        '''
        return np.linalg.norm((self.cord_x, self.cord_y, self.cord_z))

    def polar(self) -> tuple:
        """
        Restituisce le coordinate polari
        del punto in una tupla del tipo (r,theta,phi)
        """
        radius = self.magnitude()
        theta = np.arccos(self.cord_z/radius)
        phi = np.arctan(self.cord_y/self.cord_x)
        return (radius, theta, phi)

# %%
class Muon:
    def __init__(self) -> None:
        self.originpos = (0, 0, 0)
        self.theta = 0
        self.phi = 0
        self.energy = 0
        self.momentum = 0 
        self.generate()
        self.decayed = False

    def __call__(self, point: Point3D) -> Any:
        pass

    def generate(self) -> None:
        self.originpos = (rng.random(), rng.random(), rng.random())
        self.theta = rng.uniform(0.0, np.pi)
        self.phi = rng.uniform(0.0, 2 * np.pi)

    def decay(self,
              time_passed : float) -> None:
        if self.decayed:
            return None
        # vedere se decade in qualche modo
    

# %%
class Absorber:

    def __init__(self,
                 density : float,
                 atomic_number : int,
                 atomic_mass : int) -> None:
        self.density = density
        self.atomic_number = atomic_number
        self.atomic_mass = atomic_mass

    def set_position(self,
                     point: Point3D):
        self.pos_x = point.cord_x
        self.pos_y = point.cord_y
        self.pos_z = point.cord_z

    def set_dimensions(self,
                       dir_x : float,
                       dir_y : float,
                       dir_z : float) -> None:
        self.dir_x = dir_x
        self.dir_y = dir_y
        self.dir_z = dir_z

    def max_traveling_distance(self,
                               particle: Muon) -> float:
        # calcolare la distanza massima
        distance = 0
        return distance 

    def assorbing(self,
                  particle: Muon) -> None:
        
        rng.uniform(0.,self.max_traveling_distance(particle))
                
        


