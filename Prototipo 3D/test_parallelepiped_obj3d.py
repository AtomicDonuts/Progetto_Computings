''' 
test esempio per obj3d
'''
import matplotlib.pyplot as plt
import Prototipo3D as pascal


para = pascal.Parallelepiped()
para.set_position((0, 0, 0))
para.set_dimensions(1, 2, 1)


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.add_collection3d(para.obj3d()) # type: ignore

# 5. Personalizza il grafico
ax.set_xlabel("Asse X")
ax.set_ylabel("Asse Y")
ax.set_zlabel("Asse Z")  # type: ignore

# Imposta i limiti degli assi per una migliore visualizzazione
ax.set_xlim(0, 4)
ax.set_ylim(0, 4)
ax.set_zlim(0, 4)  # type: ignore

ax.set_title("Parallelepipedo con Matplotlib")

# Mostra il grafico
plt.show()
