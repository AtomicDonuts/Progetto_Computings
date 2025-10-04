from Prototipo3D import Point3D

pino = Point3D(12,3,5)
print(pino.distance_from(Point3D(14, 5, 7)))
print(pino.distance_from((14, 5, 7)))
print(pino.distance_from(14, 5, 7))
print(pino.distance_from([14, 5, 7]))
