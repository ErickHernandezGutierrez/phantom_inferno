import numpy as np
import matplotlib.pyplot as plt
import sys

npoints = int(sys.argv[1])

with open('data/PuntosElectroN%d.txt' % npoints) as file:
    lines = file.readlines()

    points = np.zeros( (npoints,3), dtype=np.float32 )

    for i,line in enumerate(lines):
        points[i,:] = np.array([float(val) for val in line.split(' ')])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(points[:,0], points[:,1], points[:,2], marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
