import numpy as np
import argparse
from utils import load_scheme

parser = argparse.ArgumentParser(description='Show scheme information.')
parser.add_argument('scheme_filename', help='input scheme filename')
args = parser.parse_args()

scheme_filename = args.scheme_filename

print('********************************************************')
print('Scheme filename: %s' % scheme_filename)
print('********************************************************')

scheme = load_scheme( scheme_filename )

bvals = {}

for (x,y,z,b) in scheme:
    bval = int(b)

    if bval in bvals:
        bvals[bval] += 1
    else:
        bvals[bval] = 1

print('Total bvals: %d' % len(bvals))
for key in bvals:
    print('├── bval=%d\t-> %d ndirs' % (key, bvals[key]))
