import numpy as np
import nibabel as nib
import itertools, sys, argparse, os
from random import random
from utils import *

parser = argparse.ArgumentParser(description='Generate phantom signal.')
parser.add_argument('study_path', help='path to the study folder')
parser.add_argument('scheme', help='protocol file in format Nx4 text file with [X Y Z b] at each row')

parser.add_argument('--template', default='templates/Phantomas', help='path to the phantom structure template. default: templates/Phantomas')
parser.add_argument('--model', default='noddi', help="model for the phantom {multi-tensor,noddi}. default: noddi")

parser.add_argument('--size_ic',  type=float, default=0.55, help='size of the intra-cellular compartment. default: 0.60')
parser.add_argument('--size_ec',  type=float, default=0.40, help='size of the extra-cellular compartment. default: 0.35')
parser.add_argument('--size_iso', type=float, default=0.05, help='size of the isotropic compartment. default: 0.05')
parser.add_argument('--select_bundles', type=int, nargs='*', help='list of the bundles to be included in the phantom. default: all')

parser.add_argument('--snr',        default=20,   type=int, help='signal to noise ratio. default: 20')
parser.add_argument('--nsubjects',  default=1,    type=int, help='number of subjects in the study. default: 1')
parser.add_argument('--ndirs',      default=5000, type=int, help='number of dispersion directions. default: 5000')
parser.add_argument('--nbatches',   default=125,  type=int, help='Number of batches. default: 125')
parser.add_argument('-noise', help='add noise to the signal', action='store_true')
parser.add_argument('-dispersion', help='add dispersion to the signal', action='store_true')
parser.add_argument('-show_dispersion', help='show dispersion directions', action='store_true')
parser.add_argument('-show_distribution', help='show distribution of the diffusivities', action='store_true')

args = parser.parse_args()

phantom = args.template
model = args.model
study = args.study_path
scheme_filename = args.scheme

if args.snr:
    SNR = args.snr
nsubjects = args.nsubjects
ndirs = args.ndirs
nbatches = args.nbatches

nbundles = phantom_info[phantom]['nbundles']
X,Y,Z = phantom_info[phantom]['dims']
nvoxels = X*Y*Z
batch_size = int(nvoxels/nbatches)

kappa = np.random.normal(loc=20, scale=1.0, size=nbundles)

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

if args.show_dispersion:
    plot_dispersion_dirs(ndirs)

if args.show_distribution:
    plot_diff_distribution()

def generate_phantom(pdds, compsize, mask, g, b, nsubjects, nvoxels, nsamples):
    print('|Generating Phantom|')

    for i in range(nsubjects):
        subject = 'sub-%.3d_ses-1' % (i+1)
        print('├── Subject %s' % subject)

        diffs = nib.load('%s/ground_truth/%s/diffs.nii.gz'%(study,subject)).get_fdata().reshape(nvoxels, 3*nbundles).astype(np.float32) # 3 diffusivities x bundle

        dwi = np.zeros( (nvoxels, nsamples), dtype=np.float32 )

        for bundle in selected_bundles:
            bundle_size = compsize[:,:,:, bundle].flatten()
            bundle_mask = mask[:,:,:, bundle].flatten()
            bundle_signal = np.zeros( (nvoxels,nsamples), dtype=np.float32 )

            for batch in range(nbatches):
                offset = batch*batch_size

                print('│   ├── Bundle %d/%d, Batch %d/%d' % (bundle+1,nbundles,batch+1,nbatches), end='\r')

                batch_pdds = pdds[offset:offset+batch_size, 3*bundle:3*bundle+3]
                batch_diffs = diffs[offset:offset+batch_size, 3*bundle:3*bundle+3]
                batch_signal = get_acquisition(model, batch_diffs, batch_pdds, g, b, kappa[i], ndirs, size_ic, size_ec, size_iso, args.dispersion)

                bundle_signal[offset:offset+batch_size, :] = batch_signal
            
            print()
            bundle_signal = (bundle_signal.transpose()*bundle_mask).transpose()
            nib.save( nib.Nifti1Image(bundle_signal.reshape(X,Y,Z,nsamples), affine, header), '%s/ground_truth/%s/bundle-%d__dwi.nii.gz'%(study,subject,bundle+1) )
            dwi += (bundle_size*bundle_signal.transpose()).transpose()

        nib.save( nib.Nifti1Image(dwi.reshape(X,Y,Z,nsamples), affine, header), '%s/ground_truth/%s/dwi.nii.gz'%(study,subject) )

        if args.noise:
            dwi = add_noise( dwi, SNR )

        nib.save( nib.Nifti1Image(dwi.reshape(X,Y,Z,nsamples), affine, header), '%s/%s/dwi.nii.gz'%(study,subject) )

# load WM masks for every bundle
mask = np.zeros( (X,Y,Z,nbundles),dtype=np.uint8 )
for bundle in range(nbundles):
    mask_filename = '%s/bundle-%d__wm-mask.nii.gz' % (phantom,bundle+1)
    mask[:,:,:, bundle] = nib.load( mask_filename ).get_fdata().astype(np.uint8)

# load PDDs for each bundle
pdds = np.zeros( (X,Y,Z,3*nbundles),dtype=np.float32 )
for bundle in range(nbundles):
    pdds_filename = '%s/bundle-%d__pdds.nii.gz' % (phantom,bundle+1)
    pdds[:,:,:, 3*bundle:3*(bundle+1)] = nib.load( pdds_filename ).get_fdata().astype(np.float32)

# load phantom wm-mask, affine and header
mask_file = nib.load('%s/wm-mask.nii.gz' % (phantom))
header = mask_file.header
affine = mask_file.affine

# load protocol
scheme = load_scheme( scheme_filename )
nsamples = len(scheme)

# create study and ground truth folders
if not os.path.exists( study ):
    os.system( 'mkdir %s'%(study) )
if not os.path.exists( '%s/ground_truth'%(study) ):
    os.system( 'mkdir %s/ground_truth'%(study) )

# calculate number of compartments
numcomp  = np.zeros( (X,Y,Z),dtype=np.uint8 )
for bundle in selected_bundles:
    numcomp += np.ones( (X,Y,Z),dtype=np.uint8 ) * mask[:,:,:, bundle]
nib.save( nib.Nifti1Image(numcomp, affine, header), '%s/ground_truth/numcomp.nii.gz'%(study) )

# calculate compartment size
compsize = np.zeros( (X,Y,Z,nbundles),dtype=np.float32 )
for (x,y,z) in itertools.product( range(X),range(Y),range(Z) ):
    if numcomp[x,y,z] > 0:
        for bundle in selected_bundles:
            compsize[x,y,z, bundle] = mask[x,y,z, bundle] / float(numcomp[x,y,z])
nib.save( nib.Nifti1Image(compsize, affine, header), '%s/ground_truth/compsize.nii.gz'%(study) )

# matrix with gradient directions
g = scheme[:,0:3]

# matrix with b-values in the diagonal
b = np.identity( nsamples,dtype=np.float32 ) * scheme[:,3]

# matrix with PDDs
pdds = pdds.reshape(nvoxels, 3*nbundles)

generate_diffs(phantom, study, affine, header, mask, nsubjects)

generate_phantom(pdds, compsize, mask, g, b, nsubjects, nvoxels, nsamples)

save_phantom_info(args, scheme, kappa, nselected)
