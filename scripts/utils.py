import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import itertools, matplotlib, os
from scipy.special import hyp1f1 as M

phantom_info = {
    'templates/Training_3D_SF': {'nbundles': 3, 'dims': [16,16,5]},
    'templates/Training_SF': {'nbundles': 5, 'dims': [16,16,5]},
    'templates/Phantomas': {'nbundles': 20, 'dims': [50,50,50]}
}

def lambdas2fa(lambdas):
    a = np.sqrt(0.5)
    b = np.sqrt( (lambdas[0]-lambdas[1])**2 + (lambdas[1]-lambdas[2])**2 + (lambdas[2]-lambdas[0])**2 )
    c = np.sqrt( lambdas[0]**2 + lambdas[1]**2 + lambdas[2]**2 )

    return a*b/c

def lambdas2md(lambdas):
    return ( lambdas[0]+lambdas[1]+lambdas[2] )/3

# Load the DWI protocol
# ------------------------------------------------------------
# It has to be in the format
#     x1 y1 z1 b1
#     ...
#     xn yn zn bn
def load_scheme(scheme_filename):
    scheme = []

    scheme_file = open(scheme_filename, 'rt')

    for line in scheme_file.readlines():
        x,y,z,b = line.split(' ')
        scheme.append( [float(x),float(y),float(z),float(b)] )

    return np.array( scheme, dtype=np.float32 )


# Give a random sample of angles between to limits
# ------------------------------------------------------------
def random_angles(min_angle, max_angle, size):
    return min_angle + (max_angle-min_angle)*np.random.rand(size[0], size[1])

# Load dispersion dirs from file
#   ndirs: int
#------------------------------------------------------------
def load_dispersion_dirs(ndirs):
    with open('data/points/PuntosElectroN%d.txt' % ndirs) as file:
        lines = file.readlines()

        dirs = np.zeros( (ndirs,3), dtype=np.float32 )

        for i,line in enumerate(lines):
            dirs[i,:] = np.array([float(val) for val in line.split(' ')])

    return dirs

# Give a set of random dispersion directions with weights
#   pdds: array (nvoxels, 3)
#   ndirs: int
#   kappa: float
# ------------------------------------------------------------
def get_dispersion(pdds, ndirs, kappa):
    nvoxels = pdds.shape[0]

    # TODO: avoid to load file several times
    dirs = load_dispersion_dirs(ndirs)

    # repeat dirs in every voxel
    dirs = np.array([dirs]*nvoxels)

    dot = np.matmul( dirs, pdds.reshape(nvoxels,3,1) )
    scale = 1/M(1.0/2.0,3.0/2.0,kappa)

    #  weights -> (nvoxels, ndirs)
    weights = scale * np.exp( kappa*(dot**2) )

    row_sums = weights.sum(axis=1)
    weights = weights / row_sums[:, np.newaxis]

    return dirs, weights.astype(np.float32)

# Compute stick signal
#   pdd: matrix (nvoxels,3)
#   d_par:  array (nvoxels)
#   g: matrix (nsamples,3)
#   b: matrix (nsamples,nsamples)
# ------------------------------------------------------------
def stick_signal(pdd, d_par, g, b):
    # (nvoxels,nsamples) <- (nvoxels,3) @ (3,nsamples)
    dot = pdd @ g.transpose()

    # (nvoxels,nsamples) <- (nvoxels) * (nvoxels,nsamples) @ (nsamples,nsamples)
    return np.exp( (-d_par*((dot**2)@b).transpose()).transpose() )

# Compute stick signal with dispersion
#   pdd: matrix (nvoxels,3)
#   d_par:  array (nvoxels)
#   g: matrix (nsamples,3)
#   b: matrix (nsamples,nsamples)
#   ndirs: int
#   kappa: int
# ------------------------------------------------------------
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

# Compute tensor signal
#   pdd: matrix (nvoxels,3)
#   d_par:  array (nvoxels)
#   d_perp: array (nvoxels)
#   g: matrix (nsamples,3)
#   b: matrix (nsamples,nsamples)
# ------------------------------------------------------------
def tensor_signal(pdd, d_par, d_perp, g, b):
    nvoxels  = pdd.shape[0]
    nsamples = g.shape[0]
    
    # (nvoxels,nsamples) <- (nvoxels,3) @ (3,nsamples)
    dot = pdd @ g.transpose()

    ones = np.ones( (nvoxels,nsamples), dtype=np.float32 )

    # (nvoxels,nsamples) <- (nvoxels)*(nvoxels,nsamples) + (nvoxels)*(nvoxels,nsamples)
    gDg = (d_perp*(ones - dot**2).transpose()).transpose() + (d_par*(dot**2).transpose()).transpose()

    # (nvoxels,nsamples) <- (nvoxels,nsamples) @ (nsamples,nsamples)
    signal = np.exp( -gDg@b )

    return signal

# Compute tensor signal with dispersion
#   pdd: matrix (nvoxels,3)
#   d_par:  array (nvoxels)
#   d_perp: array (nvoxels)
#   g: matrix (nsamples,3)
#   b: matrix (nsamples,nsamples)
#   ndirs: int
#   kappa: int
# ------------------------------------------------------------
def tensor_signal_with_dispersion(pdd, d_par, d_perp, g, b, ndirs, kappa):
    nvoxels = pdd.shape[0]
    nsamples = g.shape[0]

    dirs,weights = get_dispersion(pdd, ndirs, kappa)

    # (nvoxels, ndirs, nsamples) <- (nvoxels, ndirs, 3) @ (3, nsamples)
    dot = np.matmul(dirs, g.transpose())**2

    gDg = d_perp[:, np.newaxis, np.newaxis]*(1 - dot)
    gDg += d_par[:, np.newaxis, np.newaxis]*(dot)

    # (nvoxels,ndirs,nsamples) <- (nvoxels,ndirs,nsamples) @ (nsamples,nsamples)
    signal = np.exp( -np.matmul(gDg, b) )

    # (nvoxels,ndirs,nsamples) <- (nvoxels,ndirs,nsamples) * (nvoxels,ndirs,nsamples)
    signal *= np.repeat(weights, nsamples).reshape(nvoxels, ndirs, nsamples)

    # (nvoxels,nsamples) <- (nvoxels,ndirs,nsamples)
    signal = np.sum(signal, axis=1)

    return signal

# Generate synthetic acquisition according to the model
#   model: string
#   diff: matrix (nvoxels,3)
#   pdd: matrix (nvoxels,3)
#   g: matrix (nsamples,3)
#   b: matrix (nsamples,nsamples)
#   ndirs: int
#   kappa: int
#   size_ic: float
#   size_ec: float
#   size_iso: float
#   dispersion: bool
# ------------------------------------------------------------
def get_acquisition(model, diff, pdd, g, b, kappa, ndirs, size_ic, size_ec, size_iso, dispersion):
    nvoxels = diff.shape[0]
    nsamples = g.shape[0]

    if model == 'multi-tensor':
        d_par  = diff[:, 0]
        d_perp = diff[:, 1]

        if dispersion:
            return tensor_signal_with_dispersion(pdd, g, b, d_par, d_perp, ndirs, kappa)
        else:
            return tensor_signal(pdd, d_par, d_perp, g, b)
    elif model == 'noddi':
        d_par_ic  = diff[:, 0]
        d_par_ec  = diff[:, 1]
        d_perp_ec = diff[:, 2]

        if dispersion:
            signal_out =  size_ic * stick_signal_with_dispersion(pdd, d_par_ic, g, b, ndirs, kappa)
            signal_out += size_ec * tensor_signal_with_dispersion(pdd, d_par_ec, d_perp_ec, g, b, ndirs, kappa)
        else:
            signal_out =  size_ic * stick_signal(pdd, d_par_ic, g, b)
            signal_out += size_ec * tensor_signal(pdd, d_par_ec, d_perp_ec, g, b)

        signal_out += size_iso * np.exp( (-0.3e-3*np.ones((nvoxels,nsamples), dtype=np.float32)) @ b )

        return signal_out

# TODO: delete this function
# Generate the dMRI signal of a batch of voxels
# ------------------------------------------------------------
def generate_batch(pdd, compsize, mask, g, b, nvoxels, nsamples, offset, bsize, study, selected_bundles, nbundles, nsubjects, kappa, model, SNR, noise):
    pdd = pdd[offset:offset+bsize, :]

    signal_gt = np.zeros( (nsubjects,bsize,nsamples), dtype=np.float32 )
    signal    = np.zeros( (nsubjects,bsize,nsamples), dtype=np.float32 )

    for sub_id in range(nsubjects):
        subject = 'sub-%.3d_ses-1' % (sub_id+1)

        print('├── Subject %s' % subject)
        
        diffs = nib.load('%s/%s/ground_truth/diffs.nii.gz'%(study,subject)).get_fdata().reshape(nvoxels, 3*nbundles).astype(np.float32) # 3 diffusivities x bundle
        diffs = diffs[offset:offset+bsize, :]

        for i in selected_bundles:
            alpha = compsize[:,:,:, i].flatten()
            alpha = alpha[offset:offset+bsize]

            print('│   ├── Bundle %d...' % (i+1))

            S = get_acquisition(model, diffs[:, 3*i:3*i+3], pdd[:, 3*i:3*i+3], g, b, kappa[i])

            signal_gt[sub_id, :, :] += (alpha*S.transpose()).transpose()

        if noise:
            signal[sub_id,:,:] = add_noise( signal_gt[sub_id,:,:], SNR)
        else:
            signal[sub_id,:,:] = signal_gt[sub_id,:,:]

    return signal_gt, signal

# Add Rician noise to the signal
#   signal: matrix (nvoxels,nsamples)
#   SNR: int
# ------------------------------------------------------------
def add_noise(signal, SNR):
    nvoxels  = signal.shape[0]
    nsamples = signal.shape[1]
    sigma = 1.0 / SNR

    z = np.random.normal(loc=0, scale=1, size=(nvoxels,nsamples)) * sigma
    w = np.random.normal(loc=0, scale=1, size=(nvoxels,nsamples)) * sigma

    return np.sqrt( (signal + z)**2 + w**2 )

# Compute the success rate of a mosemap given a reference mosemap
#------------------------------------------------------------
def success_rate(ref, mosemap):
    X,Y,Z = ref.shape
    voxels = itertools.product( range(X), range(Y), range(Z) )

    success,total = 0,0

    for (x,y,z) in voxels:
        if ref[x,y,z]!=0 or mosemap[x,y,z]!=0:
            total += 1
            if ref[x,y,z] == mosemap[x,y,z]:
                success += 1

    return success / float(total)

# Generate random diffusivities in a healthy range for every bundle and subject
#------------------------------------------------------------
def generate_diffs(phantom, study, affine, header, mask, nsubjects):
    nbundles = phantom_info[phantom]['nbundles']
    X,Y,Z = phantom_info[phantom]['dims']

    d_par_ic  = np.random.normal(loc=1.5, scale=0.01, size=nsubjects*nbundles) * 1e-3
    d_par_ec  = np.random.normal(loc=1.5, scale=0.01, size=nsubjects*nbundles) * 1e-3
    d_perp_ec = np.random.normal(loc=0.6, scale=0.01, size=nsubjects*nbundles) * 1e-3

    for i in range(nsubjects):
        subject = 'sub-%.3d_ses-1' % (i+1)
        diffs = np.zeros((X,Y,Z, 3*nbundles), dtype=np.float32)

        for bundle in range(nbundles):
            diffs[:,:,:, 3*bundle]   = np.repeat( d_par_ic[i*nbundles + bundle], X*Y*Z ).reshape(X,Y,Z)  * mask[:,:,:, bundle]
            diffs[:,:,:, 3*bundle+1] = np.repeat( d_par_ec[i*nbundles + bundle], X*Y*Z ).reshape(X,Y,Z)  * mask[:,:,:, bundle]
            diffs[:,:,:, 3*bundle+2] = np.repeat( d_perp_ec[i*nbundles + bundle], X*Y*Z ).reshape(X,Y,Z) * mask[:,:,:, bundle]

        # create subject folders
        if not os.path.exists( '%s/%s' % (study,subject) ):
            os.system( 'mkdir %s/%s' % (study,subject) )
        if not os.path.exists( '%s/ground_truth/%s' % (study,subject) ):
            os.system( 'mkdir %s/ground_truth/%s' % (study,subject) )

        nib.save( nib.Nifti1Image(diffs, affine, header), '%s/ground_truth/%s/diffs.nii.gz' % (study,subject) ) 

# Plot distribution for the diffusivities
#------------------------------------------------------------
def plot_diff_distribution():
    d_par_ic  = np.random.normal(loc=1.5, scale=0.01, size=1000)
    d_par_ec  = np.random.normal(loc=1.5, scale=0.01, size=1000)
    d_perp_ec = np.random.normal(loc=0.6, scale=0.03, size=1000)
    kappa     = np.random.normal(loc=21,  scale=1.0,  size=1000)

    font = {'size' : 20}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16,9))

    ax[0,0].hist(d_par_ic,  bins=32, edgecolor='black', color='red')
    ax[0,1].hist(d_par_ec,  bins=32, edgecolor='black', color='green')
    ax[1,0].hist(d_perp_ec, bins=32, edgecolor='black', color='green')
    ax[1,1].hist(kappa,     bins=32, edgecolor='black', color='blue')

    ax[0,0].set_xlabel(r'$D_{ic}^{\parallel}$')
    ax[0,1].set_xlabel(r'$D_{ec}^{\parallel}$')
    ax[1,0].set_xlabel(r'$D_{ec}^{\perp}$')
    ax[1,1].set_xlabel(r'$Dispersion \; \kappa$')

    plt.show()

# Plot dispersion directions
#------------------------------------------------------------
def plot_dispersion_dirs(ndirs):
    dirs = load_dispersion_dirs(ndirs)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(dirs[:,0], dirs[:,1], dirs[:,2], marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # TODO: add subplot with the dispersion ODF

    plt.show()
