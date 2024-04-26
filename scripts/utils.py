import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import itertools, matplotlib, os
from scipy.special import hyp1f1 as M

# TODO: Add FiberCup phantom
phantom_info = {
    'templates/Training_3D_SF': {'nbundles': 3, 'dims': [16,16,5]},
    'templates/Training_SF': {'nbundles': 5, 'dims': [16,16,5]},
    'templates/Phantomas': {'nbundles': 20, 'dims': [50,50,50]}
}

# TODO: make this variable
mean_d_par_ic = 2.0
mean_d_par_ec = 2.0
mean_d_perp_ec = 1.0 #0.3
mean_kappa = 20
f = 0.7

def lambdas2fa(lambdas):
    a = np.sqrt(0.5)
    b = np.sqrt( (lambdas[0]-lambdas[1])**2 + (lambdas[1]-lambdas[2])**2 + (lambdas[2]-lambdas[0])**2 )
    c = np.sqrt( lambdas[0]**2 + lambdas[1]**2 + lambdas[2]**2 )

    return a*b/(c + 1e-6)

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

def load_polynomial(type='normal'):
    coefs = np.load('data/polynomials/coefs_%s.npy' % type)
    points = np.load('data/polynomials/points_%s.npy' % type)

    # Create the polynomial function
    polynomial_function = np.poly1d(coefs)

    # Adjust the polynomial output to add back the baseline
    final_polynomial = lambda x_val: polynomial_function(x_val) + np.poly1d(np.polyfit([points[:,0][0], points[:,0][-1]], [points[:,1][0], points[:,1][-1]], 1))(x_val)

    return final_polynomial

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

# Compute ball signal
#   d_iso: array (nvoxels)
#   b: matrix (nsamples,nsamples)
# ------------------------------------------------------------
def ball_signal(d_iso, b):
    nvoxels = d_iso.shape[0]
    nsamples = b.shape[0]

    return np.exp( (-np.repeat(d_iso,nsamples).reshape(nvoxels,nsamples)) @ b )

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
def get_acquisition(model, diffs, fracs, pdds, g, b, kappa, ndirs, dispersion, iso):
    nvoxels = diffs.shape[0]
    nsamples = g.shape[0]

    if model == 'multi-tensor':
        d_par  = diffs[:, 1]
        d_perp = diffs[:, 2]
        d_iso  = diffs[:, 3]

        if dispersion:
            signal_out = tensor_signal_with_dispersion(pdds, g, b, d_par, d_perp, ndirs, kappa)
        else:
            signal_out = tensor_signal(pdds, d_par, d_perp, g, b)

        if iso:
            signal_out = 0.95*signal_out + 0.05*ball_signal(d_iso, b)

    elif model == 'noddi':
        d_par_ic  = diffs[:, 0]
        d_par_ec  = diffs[:, 1]
        d_perp_ec = diffs[:, 2]
        d_iso     = diffs[:, 3]

        if dispersion:
            signal_ic = stick_signal_with_dispersion(pdds, d_par_ic, g, b, ndirs, kappa)
            signal_ec = tensor_signal_with_dispersion(pdds, d_par_ec, d_perp_ec, g, b, ndirs, kappa)
        else:
            signal_ic = stick_signal(pdds, d_par_ic, g, b)
            signal_ec = tensor_signal(pdds, d_par_ec, d_perp_ec, g, b)

        frac_ic  = np.repeat(fracs[:, 0],nsamples).reshape(nvoxels,nsamples)
        frac_ec  = np.repeat(fracs[:, 1],nsamples).reshape(nvoxels,nsamples)
        frac_iso = np.repeat(fracs[:, 2],nsamples).reshape(nvoxels,nsamples)

        signal_out = frac_ic*signal_ic + frac_ec*signal_ec

        if iso:
            signal_iso = ball_signal(d_iso, b)
            signal_out = (1-frac_iso)*signal_out + frac_iso*signal_iso

    return signal_out

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

"""
Generate volume fractions
------------------------------------------------------------
"""
def generate_fracs(phantom, study, affine, header, masks, subjects, lesion_bundles, lesion_masks, lesion_type):
    nbundles = phantom_info[phantom]['nbundles']
    X,Y,Z = phantom_info[phantom]['dims']

    for subject in subjects:
        fracs = np.zeros((X,Y,Z, 3*nbundles), dtype=np.float32)

        for bundle in range(nbundles):
            if bundle in lesion_bundles:
                fracs[:,:,:, 3*bundle]   = (masks[:,:,:, bundle]-lesion_masks[:,:,:, bundle]) * 0.65
                fracs[:,:,:, 3*bundle+1] = (masks[:,:,:, bundle]-lesion_masks[:,:,:, bundle]) * 0.35
                fracs[:,:,:, 3*bundle+2] = (masks[:,:,:, bundle]-lesion_masks[:,:,:, bundle]) * 0.05

                if lesion_type == 'demyelination':
                    fracs[:,:,:, 3*bundle]   += lesion_masks[:,:,:, bundle] * 0.55
                    fracs[:,:,:, 3*bundle+1] += lesion_masks[:,:,:, bundle] * 0.45
                    fracs[:,:,:, 3*bundle+2] += lesion_masks[:,:,:, bundle] * 0.05
                elif lesion_type == 'axonloss':
                    fracs[:,:,:, 3*bundle]   += lesion_masks[:,:,:, bundle] * 0.20
                    fracs[:,:,:, 3*bundle+1] += lesion_masks[:,:,:, bundle] * 0.80
                    fracs[:,:,:, 3*bundle+2] += lesion_masks[:,:,:, bundle] * 0.05
            else:
                fracs[:,:,:, 3*bundle]   = masks[:,:,:, bundle] * 0.65
                fracs[:,:,:, 3*bundle+1] = masks[:,:,:, bundle] * 0.35
                fracs[:,:,:, 3*bundle+2] = masks[:,:,:, bundle] * 0.05

        # create subject folders
        if not os.path.exists( '%s/%s'%(study,subject) ):
            os.system( 'mkdir %s/%s'%(study,subject) )
        if not os.path.exists( '%s/ground_truth/%s'%(study,subject) ):
            os.system( 'mkdir %s/ground_truth/%s'%(study,subject) )

        nib.save( nib.Nifti1Image(fracs,affine,header), '%s/ground_truth/%s/fracs.nii.gz'%(study,subject) ) 

"""
Generate random diffusivities in a healthy range for every bundle and subject
-----------------------------------------------------------------------------
"""
def generate_diffs(phantom, study, affine, header, masks, subjects, lesion_bundles, lesion_masks, lesion_type):
    nsubjects = len(subjects)
    nbundles = phantom_info[phantom]['nbundles']
    X,Y,Z = phantom_info[phantom]['dims']

    d_par_ic  = np.random.normal(loc=mean_d_par_ic,  scale=0.01, size=nsubjects*nbundles) * 1e-3
    d_par_ec  = np.random.normal(loc=mean_d_par_ec,  scale=0.01, size=nsubjects*nbundles) * 1e-3
    d_iso = 3e-3 # TODO: make this variable

    tortuosity_normal = load_polynomial(type='normal')
    tortuosity_lesion = load_polynomial(type=lesion_type)

    for i,subject in enumerate(subjects):
        diffs = np.zeros((X,Y,Z, 4*nbundles), dtype=np.float32)
        fracs = nib.load( '%s/ground_truth/%s/fracs.nii.gz'%(study,subject) ).get_fdata().astype(np.float32)

        for bundle in range(nbundles):
            fracs_ec  = fracs[:,:,:, 3*bundle+1].flatten()
            d_perp_ec = tortuosity_normal(fracs_ec) * 1e-3
            d_perp_ec_lesion = tortuosity_lesion(fracs_ec) * 1e-3
            
            if bundle in lesion_bundles:
                diffs[:,:,:, 4*bundle+2] = d_perp_ec.reshape(X,Y,Z) * (masks[:,:,:, bundle]-lesion_masks[:,:,:, bundle])
                diffs[:,:,:, 4*bundle+2] += d_perp_ec_lesion.reshape(X,Y,Z) * lesion_masks[:,:,:, bundle]
            else:
                diffs[:,:,:, 4*bundle+2] = d_perp_ec.reshape(X,Y,Z) * masks[:,:,:, bundle]

            diffs[:,:,:, 4*bundle]   = np.repeat( d_par_ic[i*nbundles + bundle], X*Y*Z ).reshape(X,Y,Z) * masks[:,:,:, bundle]
            diffs[:,:,:, 4*bundle+1] = np.repeat( d_par_ec[i*nbundles + bundle], X*Y*Z ).reshape(X,Y,Z) * masks[:,:,:, bundle]
            diffs[:,:,:, 4*bundle+3] = d_iso * masks[:,:,:, bundle]

        # create subject folders
        if not os.path.exists( '%s/%s'%(study,subject) ):
            os.system( 'mkdir %s/%s'%(study,subject) )
        if not os.path.exists( '%s/ground_truth/%s'%(study,subject) ):
            os.system( 'mkdir %s/ground_truth/%s'%(study,subject) )

        nib.save( nib.Nifti1Image(diffs,affine,header), '%s/ground_truth/%s/diffs.nii.gz'%(study,subject) ) 


# Plot distribution for the diffusivities
#------------------------------------------------------------
def plot_diff_distribution():
    d_par_ic  = np.random.normal(loc=mean_d_par_ic,  scale=0.01, size=1000)
    d_par_ec  = np.random.normal(loc=mean_d_par_ec,  scale=0.01, size=1000)
    d_perp_ec = np.random.normal(loc=mean_d_perp_ec, scale=0.01, size=1000)
    kappa     = np.random.normal(loc=mean_kappa,     scale=1.0,  size=1000)

    font = {'size' : 20}
    matplotlib.rc('font', **font)
    #matplotlib.rcParams['text.color'] = 'white'
    #matplotlib.rcParams['axes.labelcolor'] = 'white'
    #matplotlib.rcParams['xtick.color'] = 'white'
    #matplotlib.rcParams['ytick.color'] = 'white'
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16,9))

    #fig.set_facecolor('#212946')

    ax[0,0].set_xlabel(r'$D_{ic}^{\parallel}\;[\mu m^2/ms]$')
    ax[0,1].set_xlabel(r'$D_{ec}^{\parallel}\;[\mu m^2/ms]$')
    ax[1,0].set_xlabel(r'$D_{ec}^{\perp}\;[\mu m^2/ms]$')
    ax[1,1].set_xlabel(r'$Dispersion\;\kappa$')

    ax[0,0].set_facecolor('#EAEAF2')#('#212946')
    ax[0,1].set_facecolor('#EAEAF2')#('#212946')
    ax[1,0].set_facecolor('#EAEAF2')#('#212946')
    ax[1,1].set_facecolor('#EAEAF2')#('#212946')

    ax[0,0].set_axisbelow(True)
    ax[0,1].set_axisbelow(True)
    ax[1,0].set_axisbelow(True)
    ax[1,1].set_axisbelow(True)

    ax[0,0].grid(True, linewidth=2, color='black')
    ax[0,1].grid(True, linewidth=2, color='black')
    ax[1,0].grid(True, linewidth=2, color='black')
    ax[1,1].grid(True, linewidth=2, color='black')

    ax[0,0].hist(d_par_ic,  bins=64, color='#5A5B9F')
    ax[0,1].hist(d_par_ec,  bins=64, color='#009473')
    ax[1,0].hist(d_perp_ec, bins=64, color='#FF6F61')
    ax[1,1].hist(kappa,     bins=64, color='#F0C05A')

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

def plot_polynomials():
    poly_normal = load_polynomial()
    poly_demyelination = load_polynomial(type='demyelination')
    poly_axonloss = load_polynomial(type='axonloss')

    points_normal = np.load('data/polynomials/points_normal.npy')
    points_demyelination = np.load('data/polynomials/points_demyelination.npy')
    points_axonloss = np.load('data/polynomials/points_axonloss.npy')

    font_size=12
    font = {'size' : font_size}
    matplotlib.rc('font', **font)

    x_range = np.linspace(0.25, 1, 100)

    plt.title('Tortuosity Model Functions')

    plt.scatter(points_normal[:,0], points_normal[:,1], label='Points Normal', color='black', marker='o')
    plt.scatter(points_demyelination[:,0], points_demyelination[:,1], label='Points Demyelination', color='green', marker='x')
    plt.scatter(points_axonloss[:,0], points_axonloss[:,1], label='Points Axon Loss', color='blue', marker='^')

    plt.plot(x_range, poly_normal(x_range), label='Polynomial Normal', color='black')
    plt.plot(x_range, poly_demyelination(x_range), label='Polynomial Demyelination', color='lightgreen')
    plt.plot(x_range, poly_axonloss(x_range), label='Polynomial Axon Loss', color='lightblue')

    plt.xlabel('1-f')
    plt.ylabel(r'$D^\perp_{EC}$')

    plt.legend()
    plt.show()

# Save phantom info
#------------------------------------------------------------
def save_phantom_info(args, scheme, bundles, lesion_bundles, kappas):
    X,Y,Z = phantom_info[args.template]['dims']
    nsamples = len(scheme)

    bvals = {}
    for (x,y,z,b) in scheme:
        bval = int(b)

        if bval in bvals:
            bvals[bval] += 1
        else:
            bvals[bval] = 1

    kappas_str = ''
    for kappa in kappas:
        kappas_str += '%d ' % kappa

    bundles_str = ''
    for bundle in bundles:
        bundles_str += '%d ' % (bundle+1)

    lesion_bundles_str = ''
    for bundle in lesion_bundles:
        lesion_bundles_str += '%d ' % (bundle+1)

    bvals_str = ''
    for key in bvals:
        bvals_str += '(%d,%d) ' % (key,bvals[key])

    with open(args.study_path+'/README.txt', 'w') as file:
        file.write('│Study: %s│\n' % args.study_path)
        file.write('├── Structure: %s\n' % (args.template))
        file.write('├── Dimensions: %d x %d x %d x %d\n' % (X,Y,Z,nsamples))
        file.write('├── Voxel Size: 1 x 1 x 1 x 1 mm\n') #TODO: make this variable
        file.write('├── Model: %s\n' % (args.model))
        file.write('├── SNR: %d\n' % (args.snr))
        file.write('├── ISO: %s\n' % (args.iso))
        file.write('├── Num. Dispersion Directions: %d\n' % (args.ndirs))
        file.write('├── Num. Subjects: %d\n' % (args.nsubjects))
        file.write('├── Num. Bundles: %d\n' % (len(bundles)))
        file.write('│   └── Bundle List: %s\n' % (bundles_str))
        file.write('├── Num. Lesion Bundles: %d\n' % (len(lesion_bundles)))
        file.write('│   └── Bundle List: %s\n' % (lesion_bundles_str))
        file.write('├── Dispersion: %s\n' % (args.dispersion))
        file.write('│   ├── Num. Directions: %d\n' % (args.ndirs))
        file.write('│   └── Bundle Kappas: %s\n' % (kappas_str))
        file.write('└── Protocol: %s\n' % (args.scheme))
        file.write('    ├── Num. b-values: %d\n' % (len(bvals)))
        file.write('    ├── Num. Gradient Directions: %d\n' % (nsamples))
        file.write('    └── (b-value,ndirs): %s\n' % (bvals_str))
