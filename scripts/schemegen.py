import numpy as np
import argparse

def read_points(filename, npoints):
    points_file = open(filename, 'rt')

    lines = points_file.readlines()

    points = np.zeros((npoints,3), dtype=np.float32)

    for i in range(npoints):
        line = lines[i].split(' ')
        x,y,z = float(line[0]),float(line[1]),float(line[2])
        points[i, :] = np.array([x,y,z])

    return points

parser = argparse.ArgumentParser(description='Generate scheme.')
parser.add_argument('out_file', help='output scheme filename')
parser.add_argument('--bvals', type=int, nargs='*', help='list of the b-values')
parser.add_argument('--ndirs', type=int, nargs='*', help='list of the number of directions for each b-value')
parser.add_argument('--nb0s', type=int, default=7, help='number of b0s. [7]')
args = parser.parse_args()

out_file = args.out_file
bvals = args.bvals
ndirs = args.ndirs
nb0s = args.nb0s

scheme = []

for i in range(nb0s):
    scheme.append( np.array([0,0,0,0]) )

for (bval,n) in zip(bvals,ndirs):
    points = read_points('data/points/PuntosElectroN%d.txt'%n, n)
    for point in points:
        scheme.append( np.array([point[0], point[1], point[2], bval]) )

# save scheme to file
with open(out_file, 'wt') as scheme_file:
    for line in scheme:
        scheme_file.write('%f %f %f %f\n' % (line[0], line[1], line[2], line[3]))

    scheme_file.close()
