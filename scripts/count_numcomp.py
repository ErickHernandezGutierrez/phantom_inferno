import numpy as np
import nibabel as nib
import argparse
from utils import phantom_info

parser = argparse.ArgumentParser(description='Count number of compartments for each bundle.')
parser.add_argument('--template', default='templates/Phantomas', help='path to the phantom structure template. [templates/Phantomas]')
args = parser.parse_args()

phantom = args.template

nbundles = phantom_info[phantom]['nbundles']
X,Y,Z = phantom_info[phantom]['dims']

mask = np.zeros( (X,Y,Z,nbundles),dtype=np.uint8 )
for bundle in range(nbundles):
    mask_filename = '%s/bundle-%d__wm-mask.nii.gz' % (phantom,bundle+1)
    mask[:,:,:, bundle] = nib.load( mask_filename ).get_fdata().astype(np.uint8)

numcomp = nib.load('%s/numcomp.nii.gz' % phantom).get_fdata().astype(np.uint8)

for bundle in range(nbundles):
    N = (numcomp * mask[:,:,:, bundle]).flatten()

    unique, counts = np.unique(N, return_counts=True)
    unique = unique[1:]
    counts = counts[1:]
    #print(unique)
    #print(counts)

    data = dict(zip(unique, counts))
    total = sum(counts)
    print('Bundle-%d-------\nN=1: %.1f\nN=2: %.1f\nN=3: %.1f' % (bundle+1,counts[0]*100/total,counts[1]*100/total,sum(counts[2:])*100/total) )
