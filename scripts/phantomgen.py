import numpy as np
import nibabel as nib
import itertools, sys, argparse
from random import random
from utils import *
from scipy.special import hyp1f1 as M

parser = argparse.ArgumentParser(description='Generate phantom signal.')
parser.add_argument('phantom', help='path to the phantom structure')
parser.add_argument('model', default='multi-tensor', help='model for the phantom: multi-tensor or standard-model (default: multi-tensor)')
parser.add_argument('study_path', help='path to the study folder')
parser.add_argument('scheme', help='protocol file in format Nx4 text file with [X Y Z b] at each row')

parser.add_argument('--snr', type=int, help='signal to noise ratio (default: inf)')
parser.add_argument('--nsubjects', default=1, type=int, help='number of subjects in the study (default: 1)')
parser.add_argument('--ndirs', default=5000, type=int, help='number of dispersion directions (default: 5000)')
parser.add_argument('--select_bundles', type=int, nargs='*', help='bundles to be included in the phantom (default: all)')

args = parser.parse_args()

phantom = args.phantom
model = args.model
study = args.study_path
scheme_filename = args.scheme

if args.snr:
    SNR = args.snr
nsubjects = args.nsubjects
ndirs = args.ndirs

# number of bundles for each structure
N = {
    'structures/Training_3D_SF': 3,
    'structures/Training_SF': 5,
    'structures/Training_X': 2
}

# total number of bundles
nbundles = N[phantom]

if args.select_bundles:
    selected_bundles = args.select_bundles
    if type(selected_bundles) == int:
        selected_bundles = [selected_bundles]
    selected_bundles = [bundle-1 for bundle in selected_bundles]
else:
    selected_bundles = range(nbundles)

# number of selected bundles
nselected = len(selected_bundles)

"""
pdd: (nvoxels, 3)
ndirs: int
kappa: float
"""
def get_dispersion(pdd, ndirs, kappa):
    #p = sph2cart(1.0, azimuth, zenith)
    nvoxels = pdd.shape[0]

    # (nvoxels, 1)
    azimuth, zenith, r = cart2sph(pdd[:,0], pdd[:,1], pdd[:,2])

    azimuth = np.repeat(azimuth, ndirs).reshape(nvoxels,ndirs)
    zenith  = np.repeat(zenith,  ndirs).reshape(nvoxels,ndirs)

    eps_azimuth = random_angles(min_angle=-np.pi/2, max_angle=np.pi/2, size=(nvoxels,ndirs))
    eps_zenith  = random_angles(min_angle=-np.pi/2, max_angle=np.pi/2, size=(nvoxels,ndirs))

    dirs_azimuth = azimuth + eps_azimuth
    dirs_zenith  = zenith  + eps_zenith

    # dirs (nvoxels, ndirs, 3)
    dirs = sph2cart(np.ones((nvoxels,ndirs)), dirs_azimuth, dirs_zenith)

    dotps = np.matmul( dirs, pdd.reshape(nvoxels,3,1) )
    scale = 1/M(1.0/2.0,3.0/2.0,kappa)
    #  weights (nvoxels, ndirs)
    weights = scale * np.exp( kappa*(dotps**2) )

    return dirs, weights

def tensor_signal(pdd, d_par, d_perp, g, b):
    nvoxels  = pdd.shape[0]
    nsamples = g.shape[0]
    
    # (nvoxels,nsamples) <- (nvoxels,3) @ (3,nsamples)
    dot = pdd @ g.transpose()

    ones = np.ones( (nvoxels,nsamples) )

    # (nvoxels,nsamples) <- (nvoxels,nvoxels)@(nvoxels,nsamples) + (nvoxels,nvoxels)@(nvoxels,nsamples)
    gDg = d_perp@(ones - dot**2) + d_par@(dot**2)

    # (nvoxels,nsamples) <- (nvoxels,nsamples) @ (nsamples,nsamples)
    return np.exp( -gDg@b )

def stick_signal(pdd, d_par, g, b):
    #d_par = np.array( [d_par]*nvoxels*nsamples ).reshape(nvoxels,nsamples)

    # (nvoxels,nsamples) <- (nvoxels,3) @ (3,nsamples)
    dot = pdd @ g.transpose()

    # (nvoxels,nsamples) <- (nvoxels,nvoxels) @ (nvoxels,nsamples) @ (nsamples,nsamples)
    return np.exp( -(d_par@(dot**2))@b )

def tensor_signal_with_dispersion(pdd, g, b, lambda1, lambda2, ndirs, kappa):
    G = np.tile(g.transpose(),(nvoxels,1)).reshape(nvoxels, 3, nsamples)

    dirs, weights = get_dispersion(pdd, ndirs, kappa)

    # (nvoxels, ndirs, nsamples)
    DOT = np.matmul(dirs, G)

    UNO = np.ones((nvoxels, ndirs, nsamples))
    LAMBDA1 = UNO * lambda1
    LAMBDA2 = UNO * lambda2

    gDg = LAMBDA2*(UNO - DOT**2) + LAMBDA1*(DOT**2)

    B = np.tile(np.identity(nsamples)*b, (nvoxels,1)).reshape(nvoxels,nsamples,nsamples)

    S = np.exp( -np.matmul(gDg, B) )

    S = S * np.repeat(weights, nsamples).reshape(nvoxels, ndirs, nsamples)

    S = np.sum(S, axis=1)

    return S

""" Add Rician noise to the signal ----------------------------------------------

S: input signal with shape (nvoxels,nsamples)
SNR: signal to noise ratio
"""
def add_noise(S, SNR):
    nvoxels  = S.shape[0]
    nsamples = S.shape[1]
    sigma = 1 / SNR

    z = np.random.normal(loc=0, scale=1, size=(nvoxels,nsamples)) * sigma
    w = np.random.normal(loc=0, scale=1, size=(nvoxels,nsamples)) * sigma

    return np.sqrt( (S + z)**2 + w**2 )

numcomp_filename = '%s/numcomp.nii' % phantom
numcomp = nib.load( numcomp_filename ).get_fdata().astype(np.uint8)

scheme = load_scheme( scheme_filename )
X,Y,Z = numcomp.shape # dimensions of the phantom
nsamples = len(scheme)
nvoxels = X*Y*Z

# load WM masks
mask = np.zeros((X,Y,Z, nbundles), dtype=np.uint8)
for i in range(nbundles):
    mask[:,:,:, i] = nib.load('%s/wm-mask-%d.nii' % (phantom,i+1)).get_fdata().astype(np.uint8)

# calculate compartment size
compsize = np.zeros( (X,Y,Z, nbundles) )
numcomp  = np.zeros( (X,Y,Z), dtype=np.uint8 )
for i in selected_bundles:
    numcomp += np.ones((X,Y,Z), dtype=np.uint8) * mask[:,:,:, i]

for (x,y,z) in itertools.product( range(X), range(Y), range(Z) ):
    if numcomp[x,y,z] > 0:
        for i in selected_bundles:
            compsize[x,y,z, i] = mask[x,y,z, i] / numcomp[x,y,z]

# matrix with gradient directions
g = scheme[:,0:3]

# matrix with b-values
b = np.identity( nsamples ) * scheme[:,3]

# matrix with PDDs
pdd = nib.load( '%s/pdds.nii'%phantom ).get_fdata().reshape(nvoxels, 3*nbundles) # 3 dirs x bundle

for sub_id in range(nsubjects):
    subject = 'sub-%.3d_ses-1' % (sub_id+1)

    print('Generating signal for subject %s/' % subject)

    dwi = np.zeros( (X,Y,Z, nsamples) )
    
    lambdas_file = nib.load('%s/%s/ground_truth/lambdas.nii'%(study,subject))
    header = lambdas_file.header
    lambdas = lambdas_file.get_fdata().reshape(nvoxels, 3*nbundles) # 3 eigenvalues x bundle

    signal = np.zeros((nvoxels,nsamples))
    for i in selected_bundles:
        lambda1 = np.identity( nvoxels ) * lambdas[:, 3*i]
        lambda2 = np.identity( nvoxels ) * lambdas[:, 3*i+1]

        alpha = np.identity( nvoxels ) * compsize[:,:,:, i].flatten()
        
        S = tensor_signal(pdd[:, 3*i:3*i+3], lambda1, lambda2, g, b)
        S = alpha @ S

        signal += S

    dwi = signal.reshape(X,Y,Z, nsamples)
    dwi_file = nib.Nifti1Image(dwi, np.identity(4), header) 
    nib.save(dwi_file, '%s/%s/ground_truth/dwi.nii'% (study,subject) )

    if args.snr:
        signal = add_noise(signal, SNR=SNR)

    dwi = signal.reshape(X,Y,Z, nsamples)
    dwi_file = nib.Nifti1Image(dwi, np.identity(4), header) 
    nib.save( dwi_file, '%s/%s/dwi.nii' % (study,subject) )
