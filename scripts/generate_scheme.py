import numpy as np
import nibabel as nib

def read_points(filename):
    points_file = open(filename, 'rt')

    lines = points_file.readlines()

    ndirs = int(lines[0])

    points = np.zeros((ndirs,3), dtype=np.float32)

    for i in range(ndirs):
        x = float(lines[1 + 3*i])
        y = float(lines[1 + 3*i+1])
        z = float(lines[1 + 3*i+2])

        points[i, :] = np.array([x,y,z])

    return points

scheme = []

for i in range(7):
    scheme.append( np.array([0,0,0,0]) )

for bval in range(100, 1000, 200):
    points = read_points('../data/Elec10.txt')
    for point in points:
        scheme.append( np.array([point[0], point[1], point[2], bval]) )

for bval in range(1000, 2000, 200):
    points = read_points('../data/Elec30.txt')
    for point in points:
        scheme.append( np.array([point[0], point[1], point[2], bval]) )

for bval in range(2000, 3200, 200):
    points = read_points('../data/Elec60.txt')
    for point in points:
        scheme.append( np.array([point[0], point[1], point[2], bval]) )

scheme = np.array(scheme)

with open('scheme.txt', 'wt') as scheme_file:
    for line in scheme:
        scheme_file.write('%f %f %f %f\n' % (line[0], line[1], line[2], line[3]))

    scheme_file.close()
