import numpy as np
import nibabel as nib
import sys, itertools

mrds_results_path = sys.argv[1]

eigenvalues_filename = mrds_results_path + '/results_MRDS_Diff_BIC_EIGENVALUES.nii'
eigenvalues_file = nib.load( eigenvalues_filename )

lambdas = eigenvalues_file.get_fdata()
X,Y,Z = lambdas.shape[0:3]
voxels = itertools.product( range(X), range(Y), range(Z) )

ad = np.zeros( (X,Y,Z, 3) )
rd = np.zeros( (X,Y,Z, 3) )

for (x,y,z) in voxels:
    for fixel in range(3):
        ad[x,y,z, fixel] = lambdas[x,y,z, 3*fixel]
        rd[x,y,z, fixel] = (lambdas[x,y,z, 3*fixel+1] + lambdas[x,y,z, 3*fixel+2])/2

nib.save( nib.Nifti1Image(ad, eigenvalues_file.affine, eigenvalues_file.header), mrds_results_path + '/results_MRDS_Diff_BIC_AD.nii' )
nib.save( nib.Nifti1Image(rd, eigenvalues_file.affine, eigenvalues_file.header), mrds_results_path + '/results_MRDS_Diff_BIC_RD.nii' )
