import numpy as np
import nibabel as nib
import itertools, sys

nbundles = 20
X,Y,Z = 50,50,50

files = []
for i in range(nbundles):
    files.append(nib.load( 'bundle-%02d__wm-mask.nii.gz'%(i+1) ))

header = files[0].header
affine = files[0].affine
X,Y,Z  = files[0].shape[0:3]

masks = np.zeros((X,Y,Z,3*nbundles), dtype=np.uint8)

for i in range(nbundles):
    masks[:,:,:, 3*i] = files[i].get_fdata().astype(np.uint8)
    masks[:,:,:, 3*i+1] = files[i].get_fdata().astype(np.uint8)
    masks[:,:,:, 3*i+2] = files[i].get_fdata().astype(np.uint8)

nib.save( nib.Nifti1Image(masks, affine, header), 'aux-wm-mask.nii.gz' )

####################################################################
files = []
for i in range(nbundles):
    files.append(nib.load( 'bundle-%02d__pdds.nii.gz'%(i+1) ))

pdds = np.zeros((X,Y,Z,3*nbundles), dtype=np.float64)

for i in range(nbundles):
    pdds[:,:,:, 3*i:3*i+3] = files[i].get_fdata().astype(np.float64)

nib.save( nib.Nifti1Image(pdds, affine, header), 'pdds.nii.gz' )
