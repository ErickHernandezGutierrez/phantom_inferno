import numpy as np
import sys

in_filename = sys.argv[1]

in_file = open(in_filename, 'rt')

lines = in_file.readlines()

ndirs = int(lines[0])

points = np.zeros((ndirs,3), dtype=np.float32)

for i in range(ndirs):
    x = float(lines[1 + 3*i])
    y = float(lines[1 + 3*i+1])
    z = float(lines[1 + 3*i+2])

    points[i, :] = np.array([x,y,z])

out_filename = 'PuntosElectroN%d.txt' % ndirs

with open(out_filename, 'wt') as out_file:
    for point in points:
        out_file.write('%f %f %f\n' % (point[0], point[1], point[2]))

    out_file.close()
