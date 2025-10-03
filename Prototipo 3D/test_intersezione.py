"""
Modulo di test per l'intersezione di una linea con un parallelepipedo
"""
import Prototipo3D as lib
import numpy as np

linea = lib.Muon()
linea.originpos = lib.Point3D(0, 0, 0)
linea.theta = np.pi / 2
linea.phi = 0
linea.d_vector = linea._direction_vector()
print(linea.d_vector)

para = lib.Parallelepiped()
para.set_position((1, -0.5, -0.5))
para.set_dimensions(1, 1, 1)
if para.intersect_with_line(linea)[0]:
    print(f"{para.intersect_with_line(linea)[1][0]}\n{para.intersect_with_line(linea)[1][1]}")
