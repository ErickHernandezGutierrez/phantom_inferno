import numpy as np
import nibabel as nib
import itertools, sys, argparse
from random import random
from utils import *
from scipy.special import hyp1f1 as M

parser = argparse.ArgumentParser(description='Generate phantom signal.')
parser.add_argument('--phantom', help='path to the phantom structure')
parser.add_argument('--study_path', help='path to the study folder')
#parser.add_argument('--subject_path', help='path to the subject folder')
parser.add_argument('--scheme', help='protocol file in format Nx4 text file with [X Y Z b] at each row')
parser.add_argument('--snr', type=int, help='signal to noise ratio')
parser.add_argument('--nsubjects', default=1, type=int, help='number of subjects in the study')
#parser.add_argument('--nbundles',  default=1, type=int, help='number of bundles in the phantom')
parser.add_argument('--ndirs', default=5000, type=int, help='number of dispersion directions (default: 5000)')
parser.add_argument('--select_bundles', type=int, nargs='*', help='bundles to be included in the phantom')

args = parser.parse_args()

phantom = args.phantom

# folder of the study
study = args.study_path

scheme_filename = args.scheme
if args.snr:
    SNR = args.snr
nsubjects = args.nsubjects
ndirs = args.ndirs

N = {
    'phantoms/Training_3D_SF': 3,
    'phantoms/Training_SD_X': 2,
    'phantoms/Training_SF': 5,
    'phantoms/Training_X': 2
}

nbundles = N[phantom]

if args.select_bundles:
    selected_bundles = args.select_bundles
    if type(selected_bundles) == int:
        selected_bundles = [selected_bundles]
    selected_bundles = [bundle-1 for bundle in selected_bundles]
else:
    selected_bundles = range(nbundles)

nselected = len(selected_bundles)

# folder of the subject
#subject = sys.argv[2]

# Compute a rotation matrix corresponding to the orientation (azimuth,zenith)
#     azimuth (phi):	angle in the x-y plane
#     zenith  (theta):	angle in the x-z plane
# ---------------------------------------------------------------------------
def get_rotation(azimuth, zenith):
    azimuth = azimuth % (2*np.pi)
    zenith  = zenith % (np.pi)
    
    azimuth_rotation = np.array([
        [np.cos(azimuth), -np.sin(azimuth), 0.0],
        [np.sin(azimuth),  np.cos(azimuth), 0.0],
        [0.0,              0.0,             1.0]
    ])

    zenith_rotation = np.array([
        [ np.cos(zenith), 0.0, np.sin(zenith)],
        [ 0.0,            1.0,            0.0],
        [-np.sin(zenith), 0.0, np.cos(zenith)]
    ])

    #return np.matmul(zenith_rotation, azimuth_rotation) # primero aplica la zenith
    #return np.matmul(azimuth_rotation, zenith_rotation) # primero aplica la azimuth

    return zenith_rotation @ azimuth_rotation

def get_acquisition(voxel, scheme, noise=False, dispersion=False, sigma=0.0):
    nsamples = len(scheme)
    signal = np.zeros(nsamples)

    for i in range(nsamples):
        signal[i] = E(g=scheme[i][0:3], b=scheme[i][3], voxel=voxel, noise=noise, dispersion=dispersion)
        # todo: add noise to the signal

    return signal

def tensor_signal(g, b, pdd, lambdas):
    e = sph2cart(1, pdd[0], pdd[1])
    
    d = np.dot(g, e)

    gDg = lambdas[1]*(1 - d**2) + lambdas[0]*(d**2)

    return np.exp( -b*(gDg) )

"""
pdd: (nvoxels, 3)
ndirs: int
kappa: float
"""
def get_dispersion_vec(pdd, ndirs, kappa):
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

def vec(pdd, g, b, lambda1, lambda23):
    nvoxels  = pdd.shape[0]
    nsamples = g.shape[0]
    
    dot = pdd @ g.transpose()

    ones = np.ones( (nvoxels,nsamples) )

    gDg = lambda23@(ones - dot**2) + lambda1@(dot**2)

    return np.exp( -gDg@b )

def get_acquisition_dispersion(pdd, g, b, lambda1, lambda2, ndirs, kappa):
    G = np.tile(g.transpose(),(nvoxels,1)).reshape(nvoxels, 3, nsamples)

    dirs, weights = get_dispersion_vec(pdd, ndirs, kappa)

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

def add_noise(S, SNR):
    nvoxels  = S.shape[0]
    nsamples = S.shape[1]
    sigma = 1 / SNR

    z = np.random.normal(loc=0, scale=1, size=(nvoxels,nsamples)) * sigma
    w = np.random.normal(loc=0, scale=1, size=(nvoxels,nsamples)) * sigma

    return np.sqrt( (S + z)**2 + w**2 )

# Probe the signal at a given q-space coordinate for a given voxel configuration
# ------------------------------------------------------------------------------
def E(g, b, voxel, noise=False, dispersion=False):
    ntensors = voxel['nfascicles']
    pdds = voxel['pdds']
    R = voxel['rotmats']
    lambdas = voxel['eigenvals']
    ndirs = voxel['ndirs']
    kappa = voxel['kappa']
    alphas = voxel['fractions']
    sigma = voxel['sigma']

    signal = 0
    for i in range(ntensors):
        if dispersion:
            pdd = pdds[i]
            dirs,weights = get_dispersion(pdd[0], pdd[1], ndirs, kappa)

            dispersion_signal = 0
            for j in range(ndirs):
                dispersion_signal += weights[j]*tensor_signal(g, b, dirs[j], lambdas[i])

            signal += alphas[i]*dispersion_signal
        else:
            signal += alphas[i]*tensor_signal(g, b, pdds[i], lambdas[i])

    if noise:
        s = sigma * np.random.normal(loc=0, scale=1, size=2)
        signal = np.sqrt( (signal + s[0])**2 + s[1]**2 )

    return signal

bundle1_rotmat = np.array([
    [-0.1227878 , -0.70710678,  0.69636424],
    [-0.1227878 ,  0.70710678,  0.69636424],
    [-0.98480775,  0.        , -0.17364818]
])

bunndle2_rotmat = np.array([
    [ 0.1227878 ,  0.70710678,  0.69636424],
    [-0.1227878 ,  0.70710678, -0.69636424],
    [-0.98480775,  0.        ,  0.17364818]
])

#nbundles = 1

numcomp_filename = '%s/numcomp.nii' % phantom
numcomp = nib.load( numcomp_filename ).get_fdata().astype(np.uint8)

#mask_filename = '%s/wm-mask-1.nii' % phantom
#mask = nib.load( mask_filename ).get_fdata().astype(np.uint8)

scheme = load_scheme( scheme_filename )
X,Y,Z = numcomp.shape # dimensions of the phantom
nsamples = len(scheme)
nvoxels = X*Y*Z
#voxels = itertools.product( range(X), range(Y), range(Z) )

#if nbundles > 1:
#    compsize_filename = '%s/compsize.nii' % phantom
#else:
#    compsize_filename = '%s/wm-mask-%d.nii' % (phantom, bundles[0])
#compsize = nib.load( compsize_filename ).get_fdata().reshape( X,Y,Z, nbundles )

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

"""
if nbundles == 3:
    compsize = nib.load( compsize_filename ).get_fdata()
else:
    compsize = np.ones( (X,Y,Z,3), dtype=np.uint8 )
    compsize[:,:,:,0] *= mask
    compsize[:,:,:,1] *= masktype=int,
    compsize[:,:,:,2] *= mask
"""

#SNRs = [30, 12, np.inf]

# matrix with gradient directions
g = scheme[:,0:3]

# matrix with b-values
b = np.identity( nsamples ) * scheme[:,3]

# matrix with PDDs
pdd = nib.load( '%s/pdds.nii'%phantom ).get_fdata().reshape(nvoxels, 9)

for sub_id in range(nsubjects):
    subject = 'sub-%.3d_ses-1' % (sub_id+1)

    print('Generating signal for subject %s/' % subject)

    dwi = np.zeros( (X,Y,Z, nsamples) ) # volume for subject
    
    lambdas_file = nib.load('%s/%s/ground_truth/lambdas.nii'%(study,subject))
    header = lambdas_file.header
    #header['pixdim'][1:4] = [2,2,2]
    lambdas = lambdas_file.get_fdata().reshape(nvoxels, 3*nbundles) # eigenvalues per voxel

    signal = np.zeros((nvoxels,nsamples))
    for i in selected_bundles:
        lambda1 = np.identity( nvoxels ) * lambdas[:, 3*i]
        lambda2 = np.identity( nvoxels ) * lambdas[:, 3*i+1]

        alpha = np.identity( nvoxels ) * compsize[:,:,:, i].flatten()
        
        S = vec(pdd[:, 3*i:3*i+3], g, b, lambda1, lambda2)
        S = alpha @ S

        signal += S

    dwi = signal.reshape(X,Y,Z, nsamples)
    dwi_file = nib.Nifti1Image(dwi, np.identity(4), header) 
    #dwi_file.header['pixdim'][1:4] = [2,2,2]
    nib.save(dwi_file, '%s/%s/ground_truth/dwi.nii'% (study,subject) )

    if args.snr:
        signal = add_noise(signal, SNR=SNR)

    dwi = signal.reshape(X,Y,Z, nsamples)
    dwi_file = nib.Nifti1Image(dwi, np.identity(4), header) 
    #dwi_file.header['pixdim'][1:4] = [2,2,2]
    nib.save( dwi_file, '%s/%s/dwi.nii' % (study,subject) )
