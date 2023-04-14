from random import random
import numpy as np
from scipy.special import hyp1f1 as M

# Compute the Fractional Anisotropy (FA) of the lambdas
#------------------------------------------------------------
def lambdas2fa(lambdas):
    a = np.sqrt(0.5)
    b = np.sqrt( (lambdas[0]-lambdas[1])**2 + (lambdas[1]-lambdas[2])**2 + (lambdas[2]-lambdas[0])**2 )
    c = np.sqrt( lambdas[0]**2 + lambdas[1]**2 + lambdas[2]**2 )

    return a*b/c

# Compute the Fractional Anisotropy (FA) of the lambdas
#------------------------------------------------------------
def lambdas2md(lambdas):
    return (lambdas[0] + lambdas[1] + lambdas[2]) / 3

# Convert from spherical ccordinates to cartesian coordinates
#------------------------------------------------------------
def sph2cart(r, azimuth, zenith):
    x = r * np.cos(azimuth) * np.sin(zenith)
    y = r * np.sin(azimuth) * np.sin(zenith)
    z = r * np.cos(zenith)
    print(x.shape)

    dirs = np.zeros((x.shape[0], x.shape[1], 3))

    dirs[:,:, 0] = x
    dirs[:,:, 1] = y
    dirs[:,:, 2] = z

    #return np.array([x,y,z])
    return dirs

# Convert from cartesian coordinates to spherical ccordinates
#------------------------------------------------------------
def cart2sph(x, y, z):
    h = np.hypot(x, y)
    r = np.hypot(h, z)
    zenith = np.arctan2(h, z)
    azimuth = np.arctan2(y, x)
    
    return azimuth, zenith, r

# get Skew Symmetric Matrix From Vector
def skew_symmetric_matrix(v):
    return np.array([
        [ 0,    -v[2],  v[1]],
        [ v[2],  0,    -v[0]],
        [-v[1],  v[0],  0]
    ])

def get_rotation_from_dir(axis, dir):
    v = np.cross(axis, dir)
    c = np.dot(axis, dir)

    V = skew_symmetric_matrix(v)

    return np.identity(3) + V + V@V * (1/(1+c))

def random_angles(min_angle, max_angle, size):
    return min_angle + (max_angle-min_angle)*np.random.rand(size[0], size[1])

def random_fractions(n):
    if n==1:
        return [1]
    elif n==2:
        alpha = np.random.rand()
        return [alpha, 1-alpha]
    else:
        alpha = np.random.rand()
        beta  = np.random.rand()*alpha
        return [alpha, beta, 1-alpha-beta]

def random_eigenvals(n):
    eigenvals = []

    #TODO: make this really random
    for i in range(n):
        eigenvals.append( np.array([0.001, 0.0003, 0.0003]) )

    return eigenvals

def random_sph_dirs(ndirs):
    #azimuth = np.random.rand( ndirs ) * (2*np.pi)
    #zenith  = np.random.rand( ndirs ) * (np.pi)
    azimuth = np.ones( ndirs ) * (-np.pi/4)
    zenith  = np.ones( ndirs ) * (np.pi/2)

    dirs = sph2cart(np.ones(ndirs), azimuth, zenith)

    # transpose to return a (ndirs x 3) matrix
    return dirs.transpose()

def random_pdds(ndirs, type='sph'):
    pdds = []
    
    for i in range(ndirs):
        azimuth = np.random.rand() * (2*np.pi) # azimuth angle
        zenith  = np.random.rand() * (np.pi)   # zenith angle

        if type=='cart':
            pdds.append( sph2cart(1, azimuth, zenith) )
        elif type=='sph':
            #pdds.append( np.array([azimuth, zenith]) )
            #TODO: remove this
            pdds.append( np.array([-np.pi/4, np.pi/2]) )

    return pdds

# Load the DWI protocol
#
# It has to be in the format
#   x1 y1 z1 b1
#   ...
#   xn yn zn bn
# ------------------------------------------------------------------------------
def load_scheme(scheme_filename):
    scheme = []

    scheme_file = open(scheme_filename, 'rt')

    for line in scheme_file.readlines():
        x,y,z,b = line.split(' ')
        scheme.append( [float(x),float(y),float(z),float(b)] )

    return np.array( scheme )

# Return the dispersion weight of the dispersion direction
#   u : Principal Diffusion Direction
#   v : Dispersion Direction
#   kappa: Concentration Parameter
# ------------------------------------------------------------------------------
def get_scale(u, v, kappa):
    return (1/M(1.0/2.0,3.0/2.0,kappa)) * np.exp( kappa*((u@v)**2) )

# Return the directions and the weights of the dispersion
#   (azimuth, zenith): spherical coordinates of the PDD
#   ndirs: number of directions in the dispersion
#   kappa: concentration parameter
# ------------------------------------------------------------------------------
def get_dispersion(azimuth, zenith, ndirs, kappa):
    p = sph2cart(1.0, azimuth, zenith)

    eps_azimuth = random_angles(min_angle=-np.pi/2, max_angle=np.pi/2, size=ndirs)
    eps_zenith  = random_angles(min_angle=-np.pi/2, max_angle=np.pi/2, size=ndirs)

    dirs, weights = [], []
    for i in range(ndirs):
        dir = np.array([azimuth+eps_azimuth[i], zenith+eps_zenith[i]])

        dirs.append( dir )
        q = sph2cart(1, dir[0], dir[1])

        weight = get_scale(p, q, kappa)
        weights.append( weight )

    return np.array(dirs), np.array(weights)

def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)

def tensor2matrix(tensor):
    D = np.zeros( (3,3) )

    D[0,0] = tensor[0]
    D[1,1] = tensor[1]
    D[2,2] = tensor[2]
    D[0,1],D[1,0] = tensor[3], tensor[3]
    D[0,2],D[2,0] = tensor[4], tensor[4]
    D[1,2],D[2,1] = tensor[5], tensor[5]

    return D

def invariants(tensor):
    D = tensor2matrix(tensor)

    I1 = D[0][0] + D[1][1] + D[2][2]
    I2 = D[0][0]*D[1][1] + D[1][1]*D[2][2] + D[2][2]*D[0][0] - (D[0][1]*D[0][1] + D[0][2]*D[0][2] + D[1][2]*D[1][2])
    I3 = D[0][0]*D[1][1]*D[2][2] + 2*D[0][1]*D[0][2]*D[1][2] - (D[2][2]*D[0][1]*D[0][1] + D[1][1]*D[0][2]*D[0][2] + D[0][0]*D[1][2]*D[1][2])
    I4 = D[0][0]*D[0][0] + D[1][1]*D[1][1] + D[2][2]*D[2][2] + 2*(D[0][1]*D[0][1] + D[0][2]*D[0][2] + D[1][2]*D[1][2])

    return np.array([I1, I2, I3, I4])

def eigenvalues(tensor):
    I = invariants(tensor)

    v = (I[0]/3.0)**2 - I[1]/3.0
    s = (I[0])**3 - I[0]*I[1]/6.0 + I[2]/2.0
    o = np.arccos( clamp(s/pow(v, 3.0/2.0), min_value=-1, max_value=1) ) / 3

    lambda1 = I[0]/3 + 2*np.sqrt(v)*np.cos(o)
    lambda2 = I[0]/3 - 2*np.sqrt(v)*np.cos(np.pi/3 + o)
    lambda3 = I[0]/3 - 2*np.sqrt(v)*np.cos(np.pi/3 - o)

    return np.array([lambda1, lambda2, lambda3])

def eigenvectors(tensor):
    D = tensor2matrix(tensor)

    lambdas = eigenvalues(tensor)

    A = np.full(3, D[0,0]) - lambdas
    B = np.full(3, D[1,1]) - lambdas
    C = np.full(3, D[2,2]) - lambdas

    e = np.zeros((3,3))
    for i in range(3):
        e[i,:] = np.array([
            (D[0,1]*D[1,2] - B[i]*D[0,2])*(D[0,2]*D[1,2] - C[i]*D[0,1]),
            (D[0,2]*D[1,2] - C[i]*D[0,1])*(D[0,1]*D[0,2] - A[i]*D[1,2]),
            (D[0,1]*D[0,2] - A[i]*D[1,2])*(D[0,1]*D[1,2] - B[i]*D[0,2])
        ])

        e[i,:] = e[i,:] / np.linalg.norm(e[i,:])

    return e
