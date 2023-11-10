import numpy as np
import nibabel as nib
import itertools

masks = []
for i in range(20):
    masks.append( nib.load('bundle-%02d__wm-mask.nii.gz'%(i+1)) )

header = masks[0].header
affine = masks[0].affine
X,Y,Z  = masks[0].shape[0:3]

numcomp = np.zeros((X,Y,Z), dtype=np.uint8)

for i in range(20):
    numcomp += masks[i].get_fdata().astype(np.uint8)

nib.save( nib.Nifti1Image(numcomp, affine, header), 'numcomp.nii.gz' )
