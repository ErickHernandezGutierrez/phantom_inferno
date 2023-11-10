import numpy as np
import nibabel as nib
import argparse

parser = argparse.ArgumentParser(description='Resample 3D direction nifti image using linear interpolation')
parser.add_argument('in_dirs',  help='input direction filename')
parser.add_argument('out_dirs', help='output direction filename')
parser.add_argument('--template', help='template filename')

args = parser.parse_args()
input_filename = args.in_dirs
output_filename = args.out_dirs
if args.template:
    template_filename = args.template

input_file = nib.load( input_filename )
if args.template:
    template_file = nib.load( template_filename )

pdds = input_file.get_fdata().astype(np.float32)
X,Y,Z = pdds.shape[0:3]
output = np.zeros((2*X,2*Y,2*Z,9), dtype=np.float32)

for fixel in [0,1,2]:
    dirs = pdds[:,:,:, 3*fixel:3*fixel+3]

    #print('X')
    dx = X/float(2*X)
    nx = np.arange(2*X)
    grid = (nx + 0.5)*dx - 0.5
    tempx = np.zeros( (2*X,Y,Z, 3), dtype=np.float32 )

    for z in range(Z):
        for y in range(Y):
            for i,x in enumerate(grid):
                x1 = int(np.floor(x))
                x2 = int(np.ceil(x))

                if x1 == -1:
                    tempx[i,y,z, :] = dirs[0,y,z]
                elif x2 == X:
                    tempx[i,y,z, :] = dirs[X-1,y,z]
                else:
                    w1 = 1 - (x - x1)/(x2 - x1)
                    w2 = 1 - (x2 - x)/(x2 - x1)

                    val1 = dirs[x1,y,z]
                    val2 = dirs[x2,y,z]

                    if np.dot(val1, val2) < np.dot(val1, -val2):
                        val2 = -val2

                    tempx[i,y,z, :] = w1*val1 + w2*val2

    #print('Y')
    dy = Y/float(2*Y)
    ny = np.arange(2*Y)
    grid = (ny + 0.5)*dy - 0.5
    tempy = np.zeros( (2*X,2*Y,Z, 3), dtype=np.float32 )

    for z in range(Z):
        for x in range(2*X):
            for i,y in enumerate(grid):
                y1 = int(np.floor(y))
                y2 = int(np.ceil(y))

                if y1 == -1:
                    tempy[x,i,z, :] = tempx[x,0,z]
                elif y2 == Y:
                    tempy[x,i,z, :] = tempx[x,Y-1,z]
                else:
                    w1 = 1 - (y - y1)/(y2 - y1)
                    w2 = 1 - (y2 - y)/(y2 - y1)

                    val1 = tempx[x,y1,z]
                    val2 = tempx[x,y2,z]

                    if np.dot(val1, val2) < np.dot(val1, -val2):
                        val2 = -val2

                    tempy[x,i,z, :] = w1*val1 + w2*val2

    #print('Z')
    dz = Z/float(2*Z)
    nz = np.arange(2*Z)
    grid = (nz + 0.5)*dz - 0.5
    tempz = np.zeros( (2*X,2*Y,2*Z, 3), dtype=np.float32 )

    for x in range(2*X):
        for y in range(2*Y):
            for i,z in enumerate(grid):
                z1 = int(np.floor(z))
                z2 = int(np.ceil(z))

                if z1 == -1:
                    tempz[x,y,i, :] = tempy[x,y,0]
                elif z2 == Z:
                    tempz[x,y,i, :] = tempy[x,y,Z-1]
                else:
                    w1 = 1 - (z - z1)/(z2 - z1)
                    w2 = 1 - (z2 - z)/(z2 - z1)

                    val1 = tempy[x,y,z1]
                    val2 = tempy[x,y,z2]

                    if np.dot(val1, val2) < np.dot(val1, -val2):
                        val2 = -val2

                    tempz[x,y,i, :] = w1*val1 + w2*val2

    output[:,:,:, 3*fixel:3*fixel+3] = tempz

if args.template:
    nib.save( nib.Nifti1Image(output, template_file.affine, template_file.header), output_filename ) 
else:
    nib.save( nib.Nifti1Image(output, input_file.affine, input_file.header), output_filename ) 
