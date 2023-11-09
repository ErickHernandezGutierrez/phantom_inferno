import numpy as np
import nibabel as nib
import itertools, sys, argparse
from random import random
from utils import *
from scipy.special import hyp1f1 as M

parser = argparse.ArgumentParser(description='Generate phantom signal.')
parser.add_argument('phantom', help='path to the phantom template')
parser.add_argument('model', default='multi-tensor', help="model for the phantom {multi-tensor,noddi}. default: 'multi-tensor'")
parser.add_argument('study_path', help='path to the study folder')
parser.add_argument('scheme', help='protocol file in format Nx4 text file with [X Y Z b] at each row')

parser.add_argument('--size_ic',  type=float, default=0.7,  help='size of the intra-cellular compartment. required only for noddi')
parser.add_argument('--size_ec',  type=float, default=0.25, help='size of the extra-cellular compartment. required only for noddi')
parser.add_argument('--size_iso', type=float, default=0.05, help='size of the isotropic compartment. required only for noddi')

parser.add_argument('--snr', type=int, help='signal to noise ratio. default: inf')
parser.add_argument('--nsubjects', default=1, type=int, help='number of subjects in the study. default: 1')
parser.add_argument('--ndirs', default=250, type=int, help='number of dispersion directions. default: 250')
parser.add_argument('--vsize', type=int, default=1, help='voxel size in mm. default: 1')
parser.add_argument('--kappa', type=int, default=[24], nargs='*', help='kappa dispersion value. default: 24')
parser.add_argument('--select_bundles', type=int, nargs='*', help='bundles to be included in the phantom. default: all')
parser.add_argument('-dispersion', help='add dispersion to the signal', action='store_true')
parser.add_argument('-noise', help='add noise to the signal', action='store_true')

args = parser.parse_args()

phantom = args.phantom
model = args.model
study = args.study_path
scheme_filename = args.scheme

if args.snr:
    SNR = args.snr
nsubjects = args.nsubjects
ndirs = args.ndirs
vsize = args.vsize

# number of bundles for each structure
N = {
    'templates/Training_3D_SF': 3,
    'templates/Training_SF': 5,
    'templates/Training_X': 3,
    'templates/Phantomas': 20
}

# total number of bundles
nbundles = N[phantom]

kappa = args.kappa
if len(kappa) != nbundles:
    kappa = [kappa[0]] * nbundles

if args.select_bundles:
    selected_bundles = args.select_bundles
    if type(selected_bundles) == int:
        selected_bundles = [selected_bundles]
    selected_bundles = [bundle-1 for bundle in selected_bundles]
else:
    selected_bundles = range(nbundles)

# number of selected bundles
nselected = len(selected_bundles)

size_ic  = args.size_ic
size_ec  = args.size_ec
size_iso = args.size_iso

"""
pdd: (nvoxels, 3)
ndirs: int
kappa: float
"""
def get_dispersion(pdd, ndirs, kappa):
    nvoxels = pdd.shape[0]

    dirs = load_dispersion_dirs(ndirs)

    # repeat dirs in every voxel
    dirs = np.array([dirs]*nvoxels)

    dot = np.matmul( dirs, pdd.reshape(nvoxels,3,1) )
    scale = 1/M(1.0/2.0,3.0/2.0,kappa)

    #  weights -> (nvoxels, ndirs)
    weights = scale * np.exp( kappa*(dot**2) )

    row_sums = weights.sum(axis=1)
    weights = weights / row_sums[:, np.newaxis]

    return dirs, weights.astype(np.float32)

def tensor_signal(pdd, d_par, d_perp, g, b):
    nvoxels  = pdd.shape[0]
    nsamples = g.shape[0]
    
    # (nvoxels,nsamples) <- (nvoxels,3) @ (3,nsamples)
    dot = pdd @ g.transpose()

    ones = np.ones( (nvoxels,nsamples), dtype=np.float32 )

    # (nvoxels,nsamples) <- (nvoxels,nvoxels)@(nvoxels,nsamples) + (nvoxels,nvoxels)@(nvoxels,nsamples)
    gDg = d_perp@(ones - dot**2) + d_par@(dot**2)

    # (nvoxels,nsamples) <- (nvoxels,nsamples) @ (nsamples,nsamples)
    return np.exp( -gDg@b )

def stick_signal(pdd, d_par, g, b):
    # (nvoxels,nsamples) <- (nvoxels,3) @ (3,nsamples)
    dot = pdd @ g.transpose()

    # (nvoxels,nsamples) <- (nvoxels,nvoxels) @ (nvoxels,nsamples) @ (nsamples,nsamples)
    return np.exp( -(d_par@(dot**2))@b )

def stick_signal_with_dispersion(pdd, d_par, g, b, ndirs, kappa):
    nvoxels = pdd.shape[0]
    nsamples = g.shape[0]

    dirs, weights = get_dispersion(pdd, ndirs, kappa)

    # (nvoxels, ndirs, nsamples) <- (nvoxels, ndirs, 3) @ (3, nsamples)
    dot = np.matmul(dirs, g.transpose())**2

    # (nvoxels,ndirs,nsamples) <- (nvoxels,1,1) * (nvoxels,ndirs,nsamples) @ (nsamples,nsamples)
    S = np.exp( -d_par[:, np.newaxis, np.newaxis] * np.matmul(dot, b) )

    # (nvoxels,ndirs,nsamples) <- (nvoxels,ndirs,nsamples) * (nvoxels,ndirs,nsamples)
    S = S * np.repeat(weights, nsamples).reshape(nvoxels, ndirs, nsamples)

    # (nvoxels,nsamples) <- (nvoxels,ndirs,nsamples)
    S = np.sum(S, axis=1)

    return S

def tensor_signal_with_dispersion(pdd, g, b, lambda1, lambda2, ndirs, kappa):
    nvoxels = pdd.shape[0]
    nsamples = g.shape[0]

    dirs, weights = get_dispersion(pdd, ndirs, kappa)

    # (nvoxels, ndirs, nsamples) <- (nvoxels, ndirs, 3) @ (3, nsamples)
    dot = np.matmul(dirs, g.transpose())**2

    #uno = np.ones((nvoxels, ndirs, nsamples))

    gDg = lambda2[:, np.newaxis, np.newaxis]*(1 - dot)

    gDg += lambda1[:, np.newaxis, np.newaxis]*(dot)

    #gDg = lambda2[:, np.newaxis, np.newaxis]*(uno - dot**2) + lambda1[:, np.newaxis, np.newaxis]*(dot**2)

    # (nvoxels,ndirs,nsamples) <- (nvoxels,ndirs,nsamples) @ (nsamples,nsamples)
    S = np.exp( -np.matmul(gDg, b) )

    # (nvoxels,ndirs,nsamples) <- (nvoxels,ndirs,nsamples) * (nvoxels,ndirs,nsamples)
    S = S * np.repeat(weights, nsamples).reshape(nvoxels, ndirs, nsamples)

    # (nvoxels,nsamples) <- (nvoxels,ndirs,nsamples)
    S = np.sum(S, axis=1)

    return S

def get_acquisition(model, diff, pdd, g, b, kappa):
    nvoxels = diff.shape[0]
    nsamples = g.shape[0]

    if model == 'multi-tensor':
        d_par  = diff[:, 0]
        d_perp = diff[:, 1]

        if args.dispersion:
            return tensor_signal_with_dispersion(pdd, g, b, d_par, d_perp, ndirs, kappa)
        else:
            return tensor_signal(pdd, d_par, d_perp, g, b)
    elif model == 'noddi':
        d_par_ic  = diff[:, 0]
        d_par_ec  = diff[:, 1]
        d_perp_ec = diff[:, 2]

        if args.dispersion:
            signal_out =  size_ic * stick_signal_with_dispersion(pdd, d_par_ic, g, b, ndirs, kappa)
            signal_out += size_ec * tensor_signal_with_dispersion(pdd, g, b, d_par_ec, d_perp_ec, ndirs, kappa)
        else:
            signal_out =  size_ic * stick_signal(pdd, d_par_ic, g, b)
            signal_out += size_ec * tensor_signal(pdd, d_par_ec, d_perp_ec, g, b)

        signal_out += size_iso * np.exp( (-0.3e-3*np.ones((nvoxels,nsamples), dtype=np.float32)) @ b )

        return signal_out

""" Add Rician noise to the signal ----------------------------------------------

S: input signal with shape (nvoxels,nsamples)
SNR: signal to noise ratio
"""
def add_noise(S, SNR):
    nvoxels  = S.shape[0]
    nsamples = S.shape[1]
    sigma = 1.0 / SNR

    z = np.random.normal(loc=0, scale=1, size=(nvoxels,nsamples)) * sigma
    w = np.random.normal(loc=0, scale=1, size=(nvoxels,nsamples)) * sigma

    return np.sqrt( (S + z)**2 + w**2 )

numcomp_filename = '%s/numcomp.nii.gz' % phantom
numcomp_file = nib.load( numcomp_filename )
numcomp = numcomp_file.get_fdata().astype(np.uint8)

header = numcomp_file.header
affine = numcomp_file.affine
X,Y,Z  = numcomp.shape # dimensions of the phantom

scheme = load_scheme( scheme_filename )
nsamples = len(scheme)
nvoxels = X*Y*Z

# load WM masks
"""
mask = np.zeros((X,Y,Z, nbundles), dtype=np.uint8)
for i in range(nbundles):
    mask[:,:,:, i] = nib.load( '%s/bundle-%02d__wm-mask.nii.gz'%(phantom,i+1) ).get_fdata().astype(np.uint8)
"""
mask = nib.load( '%s/wm-mask.nii.gz'%(phantom) ).get_fdata().astype(np.uint8)

# calculate number of compartments
numcomp  = np.zeros( (X,Y,Z), dtype=np.uint8 )
for i in selected_bundles:
    numcomp += np.ones((X,Y,Z), dtype=np.uint8) * mask[:,:,:, i]
nib.save( nib.Nifti1Image(numcomp, affine, header), '%s/numcomp.nii.gz'%(study) )

# calculate compartment size
compsize = np.zeros( (X,Y,Z, nbundles), dtype=np.float32 )
for (x,y,z) in itertools.product( range(X), range(Y), range(Z) ):
    if numcomp[x,y,z] > 0:
        for i in selected_bundles:
            compsize[x,y,z, i] = mask[x,y,z, i] / float(numcomp[x,y,z])
nib.save( nib.Nifti1Image(compsize, affine, header), '%s/compsize.nii.gz'%(study) )

# matrix with gradient directions
g = scheme[:,0:3]

# matrix with b-values
b = np.identity( nsamples, dtype=np.float32 ) * scheme[:,3]

# matrix with PDDs
pdd = nib.load( '%s/pdds.nii.gz'%phantom ).get_fdata().reshape(nvoxels, 3*nbundles).astype(np.float32) # 3 dirs x bundle

for sub_id in range(nsubjects):
    subject = 'sub-%.3d_ses-1' % (sub_id+1)

    print('Generating signal for subject %s/' % subject)

    dwi = np.zeros( (X,Y,Z, nsamples), dtype=np.float32 )
    
    lambdas = nib.load('%s/%s/ground_truth/diffs.nii.gz'%(study,subject)).get_fdata().reshape(nvoxels, 3*nbundles).astype(np.float32) # 3 diffusivities x bundle

    signal = np.zeros( (nvoxels,nsamples), dtype=np.float32 )
    for i in selected_bundles:
        alpha = np.identity(nvoxels, dtype=np.float32) * compsize[:,:,:, i].flatten()

        print('\tBundle %d...' % (i+1))

        S = get_acquisition(model, lambdas[:, 3*i:3*i+3], pdd[:, 3*i:3*i+3], g, b, kappa[i])

        nib.save( nib.Nifti1Image(S.reshape(X,Y,Z, nsamples), np.identity(4)*vsize), '%s/%s/ground_truth/bundle-%d__dwi.nii.gz'%(study,subject,i+1) )

        signal += (alpha @ S)

    dwi = signal.reshape(X,Y,Z, nsamples)
    nib.save( nib.Nifti1Image(dwi, np.identity(4)*vsize), '%s/%s/ground_truth/dwi.nii.gz'%(study,subject) )

    if args.snr:
        signal = add_noise(signal, SNR=SNR)

    dwi = signal.reshape(X,Y,Z, nsamples)
    nib.save( nib.Nifti1Image(dwi, np.identity(4)*vsize), '%s/%s/dwi.nii.gz'%(study,subject) )
