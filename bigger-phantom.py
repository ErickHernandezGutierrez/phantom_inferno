import numpy as np
import nibabel as nib
import itertools

mask = [nib.load( 'upsampling/mask-1.nii' ).get_fdata().astype(np.uint8), nib.load( 'upsampling/mask-2.nii' ).get_fdata().astype(np.uint8)]
pdds = nib.load( 'gt/pdds.nii' ).get_fdata()

dirs = [ pdds[0,0,0, 0:3], pdds[0,15,0, 3:6] ]

X,Y,Z = 32,32,10

out_pdds = np.zeros( (X,Y,Z, 9) )

out_pdds[:,:,:, 0] = mask[0]*dirs[0][0]
out_pdds[:,:,:, 1] = mask[0]*dirs[0][1]
out_pdds[:,:,:, 2] = mask[0]*dirs[0][2]  
out_pdds[:,:,:, 3] = mask[1]*dirs[1][0]
out_pdds[:,:,:, 4] = mask[1]*dirs[1][1]
out_pdds[:,:,:, 5] = mask[1]*dirs[1][2]  

ones = np.ones( (X,Y,Z) )

N = mask[0] + mask[1]
alphas = np.zeros((X,Y,Z, 3))

voxels = itertools.product( range(X), range(Y), range(Z) )

for (x,y,z) in voxels:
    if mask[0][x,y,z] and mask[1][x,y,z]:
        alphas[x,y,z, 0] = 0.5
        alphas[x,y,z, 1] = 0.5
    elif mask[0][x,y,z]:
        alphas[x,y,z, 0] = 1.0
    elif mask[1][x,y,z]:
        alphas[x,y,z, 1] = 1.0

nib.save( nib.Nifti1Image(out_pdds, np.identity(4)), 'upsampling/pdds.nii' )
nib.save( nib.Nifti1Image(N, np.identity(4)), 'upsampling/numcomp.nii' )
nib.save( nib.Nifti1Image(alphas, np.identity(4)), 'upsampling/compsize.nii' )

