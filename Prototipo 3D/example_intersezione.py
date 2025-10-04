"""
Modulo di test per l'intersezione di una linea con un parallelepipedo
"""
import matplotlib.pyplot as plt
import numpy as np
import Prototipo3D as lib

linea = lib.Muon()
linea.originpos = lib.Point3D(0, 0, 0)
linea.theta = np.pi / 2
linea.phi = 0
linea.d_vector = linea._direction_vector()
t = np.linspace(-10, 10, 10)

para = lib.Parallelepiped()
para.set_position((1, -0.5, -0.5))
para.set_dimensions(1, 1, 1)
if para.intersect_with_line(linea)[0]:
    print(f"{para.intersect_with_line(linea)[1][0]}\n{para.intersect_with_line(linea)[1][1]}")

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.add_collection3d(para.obj3d(alpha_graph=0.1))  # type: ignore
ax.plot(*linea.obj3d(t), label="Retta Parametrica", color="r")
if para.intersect_with_line(linea)[0]:
    ax.scatter(*para.intersect_with_line(linea)[1][0].np_cord, color="red", s=10)
    ax.scatter(*para.intersect_with_line(linea)[1][1].np_cord, color="red", s=10)

ax.set_xlabel("Asse X")
ax.set_ylabel("Asse Y")
ax.set_zlabel("Asse Z")  # type: ignore
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)  # type: ignore
ax.grid()
plt.show()
