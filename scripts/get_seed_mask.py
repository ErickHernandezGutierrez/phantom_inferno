import numpy as np
import nibabel as nib
import sys

filename = sys.argv[1]
out_filename = sys.argv[2]
mask_filename = sys.argv[3]

file = nib.load( filename )
data = file.get_fdata()
mask = nib.load( mask_filename ).get_fdata()

cmax = np.max( data )
cmin = np.min( data )

data = 1 - (data - cmin)/(cmax - cmin)

data *= mask

nib.save( nib.Nifti1Image(data, file.affine, file.header), out_filename )
