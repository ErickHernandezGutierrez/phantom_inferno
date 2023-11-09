import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import sys, itertools, argparse, os

def lambdaFA2lambdaMD(lambda1, FA):
    a = FA / np.sqrt(3.0 - 2.0*FA*FA)

    MD = lambda1 / (1.0 + 2.0*a)
    lambda23 = MD * (1.0 - a)

    return lambda23

def lambdas2fa(lambda1, lambda23):
    a = np.sqrt(0.5)
    b = np.sqrt( (lambda1-lambda23)**2 + (lambda23-lambda23)**2 + (lambda23-lambda1)**2 )
    c = np.sqrt( lambda1**2 + lambda23**2 + lambda23**2 )
    
    return a*b/c

def lambdas2md(lambda1, lambda2):
    return (lambda1+lambda2+lambda2)/3.0

parser = argparse.ArgumentParser(description='Generate phantom diffusivities.')
parser.add_argument('phantom', default='Training_3D_SF', help="phantom structure {Training_3D_SF,Training_SF,Training_X,Phantomas}. default: 'Training_3D_SF'")
parser.add_argument('model', default='multi-tensor', help="model for the phantom {multi-tensor,noddi}. default: 'multi-tensor'")
parser.add_argument('study_path', help='path to the study folder')
parser.add_argument('--nsubjects', type=int, default=1, help='number of subjects to generate. default: 1')
parser.add_argument('--vsize', type=int, default=1, help='voxel size in mm of the phantom. default: 1')
parser.add_argument('--lesion_mask', nargs='*', help='lesion mask for each bundle')

parser.add_argument('--healthy_lambdas', type=float, default=[2e-3, 0.5e-3], nargs=2, help='diffusivities for healthy tissue. default: [2e-3, 0.5e-3] mm^2/s')
parser.add_argument('--damaged_lambdas', type=float, default=[2e-3, 0.5e-3], nargs=2, help='diffusivities for damaged tissue. default: [2e-3, 0.5e-3] mm^2/s')

parser.add_argument('--d_par_ic',  type=float, default=[2e-3], nargs='+', help='IC parallel diffusivities. default: 2e-3 mm^2/s')
parser.add_argument('--d_par_ec',  type=float, default=[2e-3], nargs='+', help='EC parallel diffusivity. default: 2e-3 mm^2/s')
parser.add_argument('--d_perp_ec', type=float, default=[0.5e-3], nargs='+', help='EC perpendicular diffusivity. default: 0.5e-3 mm^2/s')
parser.add_argument('--d_iso',     type=float, default=0.3e-3, help='ISO diffusivity. default: 0.3e-3 mm^2/s')

parser.add_argument('--lesion_d_par_ic',  type=float, default=[2e-3], nargs='+', help='IC parallel diffusivities. default: 2e-3 mm^2/s')
parser.add_argument('--lesion_d_par_ec',  type=float, default=[2e-3], nargs='+', help='EC parallel diffusivity. default: 2e-3 mm^2/s')
parser.add_argument('--lesion_d_perp_ec', type=float, default=[0.5e-3], nargs='+', help='EC perpendicular diffusivity. default: 0.5e-3 mm^2/s')
parser.add_argument('--lesion_d_iso',     type=float, default=0.3e-3, help='ISO diffusivity. default: 0.3e-3 mm^2/s')

parser.add_argument('--healthy_mean', type=float, nargs=2, help='mean of diffusivities for healthy tissue')
parser.add_argument('--healthy_var',  type=float, nargs=2, help='variance of diffusivities for healthy tissue')
parser.add_argument('--lesion_mean',  type=float, nargs=2, help='mean of diffusivities for healthy tissue')
parser.add_argument('--lesion_var',   type=float, nargs=2, help='variance of diffusivities for healthy tissue')
parser.add_argument('-show_hist',  help='If True, show the histograms of the generated diffusivities', action='store_true')

args = parser.parse_args()

phantom = args.phantom
study = args.study_path
model = args.model

nsubjects = args.nsubjects
vsize = args.vsize

N = {
    'templates/Training_3D_SF': 3,
    'templates/Training_SF': 5,
    'templates/Training_X': 3,
    'templates/Phantomas': 20
}

dims = {
    'templates/Training_3D_SF': [16,16,5],
    'templates/Training_SF': [16,16,5],
    'templates/Training_X': [32,32,10],
    'templates/Phantomas': [50,50,50]
}

lesion_mask_filenames = args.lesion_mask

nbundles = N[phantom]

X,Y,Z = dims[phantom]

d_par_ic = args.d_par_ic
if len(d_par_ic) == 1:
    d_par_ic = np.repeat( d_par_ic, nbundles*nsubjects*X*Y*Z ).reshape(nbundles, nsubjects, X,Y,Z)
elif len(d_par_ic) == nbundles:
    d_par_ic = np.repeat( d_par_ic, nsubjects*X*Y*Z ).reshape(nbundles, nsubjects, X,Y,Z)

d_par_ec = args.d_par_ec
if len(d_par_ec) == 1:
    d_par_ec = np.repeat( d_par_ec, nbundles*nsubjects*X*Y*Z ).reshape(nbundles, nsubjects, X,Y,Z)
elif len(d_par_ec) == nbundles:
    d_par_ec = np.repeat( d_par_ec, nsubjects*X*Y*Z ).reshape(nbundles, nsubjects, X,Y,Z)

d_perp_ec = np.array( args.d_perp_ec )
if len(d_perp_ec) == 1:
    d_perp_ec = np.repeat( d_perp_ec, nbundles*nsubjects*X*Y*Z ).reshape(nbundles, nsubjects, X,Y,Z)
elif len(d_perp_ec) == nbundles:
    d_perp_ec = np.repeat( d_perp_ec, nsubjects*X*Y*Z ).reshape(nbundles, nsubjects, X,Y,Z)

d_iso     = args.d_iso

# Load WM and lesion masks
mask = np.zeros((X,Y,Z, nbundles), dtype=np.uint8)
lesion_mask = np.zeros((X,Y,Z, nbundles), dtype=np.uint8)
for i in range(nbundles):
    mask[:,:,:, i]        = nib.load('%s/bundle-%.2d__wm-mask.nii.gz'        % (phantom,i+1)).get_fdata()
    lesion_mask[:,:,:, i] = nib.load('%s/bundle-%.2d__no-lesion-mask.nii.gz' % (phantom,i+1)).get_fdata()

# load header and affine matrix
temp = nib.load('%s/bundle-01__wm-mask.nii.gz' % (phantom))
header = temp.header
affine = temp.affine

if not os.path.exists( study ):
    os.system( 'mkdir %s' % study )

for sub_id in range(nsubjects):
    subject = '%s/sub-%.3d_ses-1' % (study,sub_id+1)

    print('generating lambdas for %s from phantom %s' % (subject,phantom))

    diffusivities = np.zeros((X,Y,Z, 3*nbundles))

    for bundle_id in range(nbundles):
        diffusivities[:,:,:, 3*bundle_id]   += d_par_ic[bundle_id, sub_id, :,:,:]  * (mask[:,:,:, bundle_id]-lesion_mask[:,:,:, bundle_id])
        diffusivities[:,:,:, 3*bundle_id+1] += d_par_ec[bundle_id, sub_id, :,:,:]  * (mask[:,:,:, bundle_id]-lesion_mask[:,:,:, bundle_id])
        diffusivities[:,:,:, 3*bundle_id+2] += d_perp_ec[bundle_id, sub_id, :,:,:] * (mask[:,:,:, bundle_id]-lesion_mask[:,:,:, bundle_id])

        diffusivities[:,:,:, 3*bundle_id]   += d_par_ic[bundle_id, sub_id, :,:,:]  * (lesion_mask[:,:,:, bundle_id])
        diffusivities[:,:,:, 3*bundle_id+1] += d_par_ec[bundle_id, sub_id, :,:,:]  * (lesion_mask[:,:,:, bundle_id])
        diffusivities[:,:,:, 3*bundle_id+2] += d_perp_ec[bundle_id, sub_id, :,:,:] * (lesion_mask[:,:,:, bundle_id])

    # create subject folders
    if not os.path.exists( subject ):
        os.system( 'mkdir %s' % (subject) )
    if not os.path.exists( '%s/ground_truth' % (subject) ):
        os.system( 'mkdir %s/ground_truth' % (subject) )

    # save diffusivities
    nib.save( nib.Nifti1Image(diffusivities, affine*vsize, header), '%s/ground_truth/diffs.nii.gz'%(subject) ) 
