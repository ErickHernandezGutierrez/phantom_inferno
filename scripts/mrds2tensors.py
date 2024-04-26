import numpy as np
import nibabel as nib
import itertools, sys, argparse

parser = argparse.ArgumentParser( description='Convert MRDS output to tensors.' )
parser.add_argument('results_mrds', help='MRDS results folder')
#parser.add_argument('tensors_out', help='output tensor filename')
parser.add_argument('--method', default='Diff', help='estimation method: Diff, Equal or Fixel. [Diff]')
parser.add_argument('--modsel', default='BIC', help='model selector. [BIC]')
parser.add_argument('--prefix', default='results', help='prefix of the MRDS results. [results]')
parser.add_argument('--mask', help='optional mask file')
parser.add_argument('-inv', help='invert axis x', action='store_true')
args = parser.parse_args()

results_mrds = args.results_mrds
prefix = args.prefix
method = args.method
modsel = args.modsel

EIGENVALUES_FILENAME = '%s/%s_MRDS_%s_%s_EIGENVALUES.nii.gz' % (results_mrds,prefix,method,modsel)
PDDS_CARTESIAN_FILENAME = '%s/%s_MRDS_%s_%s_PDDs_CARTESIAN.nii.gz' % (results_mrds,prefix,method,modsel)
NUMCOMP_FILENAME = '%s/%s_MRDS_%s_%s_NUM_COMP.nii.gz' % (results_mrds,prefix,method,modsel)
OUTPUT_FILENAME = ['%s/%s_MRDS_%s_%s_TENSOR_T%d.nii.gz' % (results_mrds,prefix,method,modsel,i) for i in range(3)]

eigenvalues_file = nib.load(EIGENVALUES_FILENAME)
pdds_file        = nib.load(PDDS_CARTESIAN_FILENAME)
numcomp_file     = nib.load(NUMCOMP_FILENAME)

eigenvalues = eigenvalues_file.get_fdata()
pdds        = pdds_file.get_fdata()
N           = numcomp_file.get_fdata().astype(np.uint8)

X,Y,Z = N.shape
voxels = itertools.product( range(X),range(Y),range(Z) )

if args.mask:
    MASK_FILENAME = args.mask
    mask = nib.load(MASK_FILENAME).get_fdata().astype(np.uint8)
else:
    mask = np.ones((X,Y,Z), dtype=np.uint8)

def skew_symmetric_matrix(v):
    return np.array([
        [ 0,    -v[2],  v[1]],
        [ v[2],  0,    -v[0]],
        [-v[1],  v[0],  0]
    ])

# a -> reference axis
def get_rotation_from_dir(a, dir):
    #a = (0,1,0)
    v = np.cross(a, dir)
    c = np.dot(a, dir)

    V = skew_symmetric_matrix(v)

    return np.identity(3) + V + V@V * (1/(1+c))

tensors = np.zeros( (X,Y,Z,6, 3) )

for (x,y,z) in voxels:
    if mask[x,y,z]:
        for i in range( N[x,y,z] ):
            dir     = pdds[x,y,z, 3*i:3*i+3]
            lambdas = eigenvalues[x,y,z, 3*i:3*i+3]

            if args.inv:
                R = get_rotation_from_dir((1,0,0), (-dir[0], dir[1], dir[2])) # inverting x coord
            else:
                R = get_rotation_from_dir((1,0,0), ( dir[0], dir[1], dir[2]))

            T = R.transpose() @ np.diag(lambdas) @ R

            tensors[x,y,z, :, i] = np.array([ T[0,0], T[1,1], T[2,2], T[0,1], T[0,2], T[1,2] ])

nib.save( nib.Nifti1Image(tensors[:,:,:,:, 0], numcomp_file.affine, numcomp_file.header), OUTPUT_FILENAME[0] )
nib.save( nib.Nifti1Image(tensors[:,:,:,:, 1], numcomp_file.affine, numcomp_file.header), OUTPUT_FILENAME[1] )
nib.save( nib.Nifti1Image(tensors[:,:,:,:, 2], numcomp_file.affine, numcomp_file.header), OUTPUT_FILENAME[2] )
