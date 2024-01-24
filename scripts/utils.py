import numpy as np
import nibabel as nib
import itertools, os

total_bundles = {
    'templates/Training_3D_SF': 3,
    'templates/Training_SF': 5,
    'templates/Phantomas': 20
}

dims = {
    'templates/Training_3D_SF': [16,16,5],
    'templates/Training_SF': [16,16,5],
    'templates/Phantomas': [50,50,50]
}

phantom_info = {
    'templates/Training_3D_SF': {'nbundles': 3, 'dims': [16,16,5]},
    'templates/Training_SF': {'nbundles': 5, 'dims': [16,16,5]},
    'templates/Phantomas': {'nbundles': 20, 'dims': [50,50,50]}
}

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

# Convert from spherical ccordinates to cartesian coordinates
# ------------------------------------------------------------
def sph2cart(r, azimuth, zenith):
    x = r * np.cos(azimuth) * np.sin(zenith)
    y = r * np.sin(azimuth) * np.sin(zenith)
    z = r * np.cos(zenith)

    dirs = np.zeros((x.shape[0], x.shape[1], 3))

    dirs[:,:, 0] = x
    dirs[:,:, 1] = y
    dirs[:,:, 2] = z

    return dirs

# Convert from cartesian coordinates to spherical ccordinates
#------------------------------------------------------------
def cart2sph(x, y, z):
    h = np.hypot(x, y)
    r = np.hypot(h, z)
    zenith = np.arctan2(h, z)
    azimuth = np.arctan2(y, x)
    
    return azimuth, zenith, r

# Load dispersion dirs from file
#------------------------------------------------------------
def load_dispersion_dirs(ndirs):
    with open('data/points/PuntosElectroN%d.txt' % ndirs) as file:
        lines = file.readlines()

        dirs = np.zeros( (ndirs,3), dtype=np.float32 )

        for i,line in enumerate(lines):
            dirs[i,:] = np.array([float(val) for val in line.split(' ')])

    return dirs

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

    d_par_ic  = np.random.uniform(low=1.8,  high=2.0, size=nsubjects*nbundles) * 1e-3
    d_par_ec  = np.random.uniform(low=1.8,  high=2.0, size=nsubjects*nbundles) * 1e-3
    d_perp_ec = np.random.uniform(low=0.5, high=0.6, size=nsubjects*nbundles) * 1e-3

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
        if not os.path.exists( '%s/%s/ground_truth' % (study,subject) ):
            os.system( 'mkdir %s/%s/ground_truth' % (study,subject) )
        nib.save( nib.Nifti1Image(diffs, affine, header), '%s/%s/ground_truth/diffs.nii.gz' % (study,subject) ) 
