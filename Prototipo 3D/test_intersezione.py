"""
Modulo di test per l'intersezione di una linea con un parallelepipedo
"""
import Prototipo3D as lib
import numpy as np

linea = lib.Muon()
linea.originpos = lib.Point3D(0, 0, 0)
linea.phi = np.pi/2
linea.theta = 0
linea.d_vector = linea._direction_vector()
print(linea.d_vector)

para = lib.Paralleogram()
para.set_position((1, -0.5, -0.5))
para.set_dimensions(1, 1, 1)
para.intersect_with_line(linea)
