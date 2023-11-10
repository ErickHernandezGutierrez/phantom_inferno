import numpy as np
import nibabel as nib
import scipy.io, sys, argparse

parser = argparse.ArgumentParser(description='Convert .mat file to .nii image.')
parser.add_argument('mat_filename', help='input mat filename')
parser.add_argument('nii_filename', help='output nii filename')
parser.add_argument('key', help='key field of the mat file')

args = parser.parse_args()

mat_filename = args.mat_filename
nii_filename = args.nii_filename
key = args.key

mat = scipy.io.loadmat( mat_filename )

data = mat[key]

nib.save( nib.Nifti1Image(data, np.identity(4)*2), nii_filename )
