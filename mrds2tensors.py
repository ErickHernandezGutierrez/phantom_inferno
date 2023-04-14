import numpy as np
import nibabel as nib
import itertools, sys
from utils import *

EIGENVALUES_FILENAME    = sys.argv[1]
COMP_SIZE_FILENAME      = sys.argv[2]
PDDS_CARTESIAN_FILENAME = sys.argv[3]
OUTPUT_FILENAME         = sys.argv[4]

eigenvalues_img = nib.load(EIGENVALUES_FILENAME)
sizes_img       = nib.load(COMP_SIZE_FILENAME)
pdds_img        = nib.load(PDDS_CARTESIAN_FILENAME)

eigenvalues = eigenvalues_img.get_fdata()
sizes       = sizes_img.get_fdata()
pdds        = pdds_img.get_fdata()

if len(sys.argv) > 5:
    MASK_FILENAME = sys.argv[5]
    mask = nib.load(MASK_FILENAME).get_fdata().astype(np.uint8)
else:
    MASK_FILENAME = None
    mask = np.ones( eigenvalues.shape[0:3] )

X,Y,Z = eigenvalues.shape[0:3]
voxels = itertools.product( range(X), range(Y), range(Z) )

tensors = np.zeros( (X,Y,Z,6, 3) )

for (x,y,z) in voxels:
    if mask[x,y,z] > 0:
        N = sum(sizes[x,y,z, :]>0)
        
        for i in range(N):
            dir     = pdds[x,y,z, 3*i:3*i+3]
            lambdas = eigenvalues[x,y,z, 3*i:3*i+3]
            alpha   = sizes[x,y,z, i]

            R = get_rotation_from_dir((1,0,0), (-dir[0] ,dir[1], dir[2]))
            T = R.transpose()@np.diag(lambdas)@R

            tensors[x,y,z, :, i] = np.array([ T[0,0], T[1,1], T[2,2], T[0,1], T[0,2], T[1,2] ])

nib.save( nib.Nifti1Image(tensors[:,:,:,:, 0], sizes_img.affine, sizes_img.header), OUTPUT_FILENAME[:-4]+'_T0.nii' )
nib.save( nib.Nifti1Image(tensors[:,:,:,:, 1], sizes_img.affine, sizes_img.header), OUTPUT_FILENAME[:-4]+'_T1.nii' )
nib.save( nib.Nifti1Image(tensors[:,:,:,:, 2], sizes_img.affine, sizes_img.header), OUTPUT_FILENAME[:-4]+'_T2.nii' )
