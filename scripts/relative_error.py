import numpy as np
import nibabel as nib
import argparse
from utils import phantom_info

parser = argparse.ArgumentParser(description='Count number of compartments for each bundle.')
parser.add_argument('results_tractometry', help='path to the tractometry results')
parser.add_argument('results_ground_truth', help='path to the ground truth results')
parser.add_argument('results_dti', help='path to the ground truth results')
parser.add_argument('--template', default='templates/Phantomas', help='path to the phantom structure template. [templates/Phantomas]')
args = parser.parse_args()

results_tractometry = args.results_tractometry
results_ground_truth = args.results_ground_truth
results_dti = args.results_dti
phantom = args.template

nbundles = phantom_info[phantom]['nbundles']
X,Y,Z = phantom_info[phantom]['dims']

# load WM masks
mask = np.zeros( (X,Y,Z,nbundles),dtype=np.uint8 )
for bundle in range(nbundles):
    mask_filename = '%s/bundle-%d__wm-mask.nii.gz' % (phantom,bundle+1)
    mask[:,:,:, bundle] = nib.load( mask_filename ).get_fdata().astype(np.uint8)

for bundle in range(nbundles):
    print('Bundle-%d---------'%(bundle+1))
    for metric in ['fa','md','rd','ad']:
        data = nib.load( '%s/sub-001_ses-1/Fixel_MRDS/bundle-%d_ic_%s_metric.nii.gz'%(results_tractometry,bundle+1,metric) ).get_fdata()*mask[:,:,:, bundle]
        dti = nib.load( '%s/results_DTInolin_%s.nii.gz'%(results_dti,metric.upper()) ).get_fdata()*mask[:,:,:, bundle]
        gt = nib.load( '%s/sub-001_ses-1/bundle-%d__%s.nii.gz'%(results_ground_truth,bundle+1,metric) ).get_fdata()*mask[:,:,:, bundle]

        mrds_err = np.abs(data-gt) / gt
        mrds_err = mrds_err.flatten()
        mrds_err = mrds_err[np.isnan(mrds_err) == False]

        dti_err = np.abs(dti-gt) / gt
        dti_err = dti_err.flatten()
        dti_err = dti_err[np.isnan(dti_err) == False]

        print('%s: %.1f' % (metric.upper(),np.mean(dti_err)*100))
        print('fixel-%s: %.1f' % (metric.upper(),np.mean(mrds_err)*100))

"""
in_data = nib.load( in_filename ).get_fdata().flatten()
in_data = in_data[in_data > 0]

relative_err = np.mean(np.abs( (in_data-gt)/gt )) * 100

print('Relative Error = ', relative_err)
"""
