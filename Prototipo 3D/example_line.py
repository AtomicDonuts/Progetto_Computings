'''
Test visivo per la direzionalità delle Line
'''
import Prototipo3D as lib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

linea_z = lib.Muon()
linea_z.originpos = lib.Point3D(0, 0, 0)
linea_z.theta = 0.0
linea_z.phi = 0.0
linea_z.d_vector = linea_z._direction_vector()  

linea_x = lib.Muon()
linea_x.originpos = lib.Point3D(0, 0, 0)
linea_x.theta = np.pi / 2
linea_x.phi = 0.0
linea_x.d_vector = linea_x._direction_vector()  

linea_y = lib.Muon()
linea_y.originpos = lib.Point3D(0, 0, 0)
linea_y.theta = np.pi / 2
linea_y.phi = np.pi / 2
linea_y.d_vector = linea_y._direction_vector()
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.quiver(
    linea_z.originpos[0],
    linea_z.originpos[1],
    linea_z.originpos[2],
    linea_z.d_vector[0],
    linea_z.d_vector[1],
    linea_z.d_vector[2],
    arrow_length_ratio=0.1,
    color = "red",
    label="Vettore Z",
)
ax.quiver(
    linea_x.originpos[0],
    linea_x.originpos[1],
    linea_x.originpos[2],
    linea_x.d_vector[0],
    linea_x.d_vector[1],
    linea_x.d_vector[2],
    arrow_length_ratio=0.1,
    color="green",
    label="Vettore X",
)
ax.quiver(
    linea_y.originpos[0],
    linea_y.originpos[1],
    linea_y.originpos[2],
    linea_y.d_vector[0],
    linea_y.d_vector[1],
    linea_y.d_vector[2],
    arrow_length_ratio=0.1,
    color="blue",
    label="Vettore Y",
)
ax.set_xlabel("Asse X")
ax.set_ylabel("Asse Y")
ax.set_zlabel("Asse Z") # type: ignore
ax.set_xlim(-1,1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1) # type: ignore
ax.legend()
ax.grid(True)
plt.show()
