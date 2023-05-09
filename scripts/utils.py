import numpy as np


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

    return np.array( scheme )


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
    with open('data/PuntosElectroN%d.txt' % ndirs) as file:
        lines = file.readlines()

        dirs = np.zeros( (ndirs,3), dtype=np.float32 )

        for i,line in enumerate(lines):
            dirs[i,:] = np.array([float(val) for val in line.split(' ')])

    return dirs
