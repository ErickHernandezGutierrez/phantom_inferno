#!/usr/bin/python
import os,argparse,matplotlib
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Plot MRDS estimated metrics histograms')
parser.add_argument('input_path', help='input path to the estimated MRDS metrics')
parser.add_argument('--prefix', default='results', help='MRDS prefix. [results]')
parser.add_argument('--modsel', default='bic', help='MRDS model selector (aic,bic,ftest). [bic]')
parser.add_argument('--mask', help='mask filename')
parser.add_argument('--bins', default=32, type=int, help='number of bins of the histograms. [64]')
parser.add_argument('--dti', help='path to the estimated DTI metrics')
parser.add_argument('--ground_truth', help='path to the ground truth')
args = parser.parse_args()

results_mrds = args.input_path
results_dti = args.dti
results_gt = args.ground_truth
prefix = args.prefix
modsel = args.modsel
mask_filename = args.mask
bins = args.bins

metrics = ['FA','MD','AD','RD']
#metrics = ['fixel-FA','fixel-MD','fixel-AD','fixel-RD']
coords  = [0,1,2,3]
font_size = 15

#background_color = '#202946'
#background_color = '#D1D2D3'
background_color = '#292C35'
#ground_truth_color = '#000000'
ground_truth_color = '#A0288D'
color = {
    'FA': '#0E7D48',
    'MD': '#107B7F',
    'RD': '#172E73',
    'AD': '#431B6B',
    'fixel-FA': '#AD151D',
    'fixel-MD': '#AE3C21',
    'fixel-RD': '#B77A21',
    'fixel-AD': '#C0B324'
}
facecolor = {
    'FA': '#9CD1B6',
    'MD': '#97D1D3',
    'RD': '#9A9CCB',
    'AD': '#AC9AC7',
    'fixel-FA': '#F6AA92',
    'fixel-MD': '#F8B89A',
    'fixel-RD': '#FED6A3',
    'fixel-AD': '#FFF7B0'
}

data = {
    'FA': [],
    'MD': [],
    'AD': [],
    'RD': [],
    'fixel-FA': [],
    'fixel-MD': [],
    'fixel-AD': [],
    'fixel-RD': []
}

ground_truth = {
    'FA': [],
    'MD': [],
    'AD': [],
    'RD': []
}

if args.mask:
    mask = nib.load(mask_filename).get_fdata().astype(np.uint8)

for metric in ['FA','MD','AD','RD']:
    metric_mrds = nib.load( '%s/%s_MRDS_Diff_%s_%s.nii.gz'%(results_mrds,prefix,modsel.upper(),metric) ).get_fdata()
    
    if args.mask:
        metric_mrds[:,:,:, 0] *= mask
        metric_mrds[:,:,:, 1] *= mask
        metric_mrds[:,:,:, 2] *= mask

    if metric in ['MD','AD','RD']:
        metric_mrds *= 1e3

    metric_mrds = metric_mrds[metric_mrds != 0]
    data['fixel-'+metric] = metric_mrds.flatten()

    if args.dti:
        metric_dti = nib.load( '%s/%s_DTInolin_%s.nii.gz'%(results_dti,prefix,metric) ).get_fdata()
        
        if args.mask:
            metric_dti *= mask

        if metric in ['MD','AD','RD']:
            metric_dti *= 1e3

        metric_dti = metric_dti[metric_dti != 0]
        data[metric] = metric_dti.flatten()

    if args.ground_truth:
        metric_bundles = np.array([])
        nbundles = 3
        for i in range(nbundles):
            metric_bundle = nib.load( '%s/bundle-%d__%s.nii.gz'%(results_gt,i+1,metric.lower()) ).get_fdata()

            if args.mask:
                metric_bundle *= mask

            if metric in ['MD','AD','RD']:
                metric_bundle *= 1e3

            metric_bundle = metric_bundle[metric_bundle > 0]
            metric_bundles = np.concatenate((metric_bundles, metric_bundle), axis=None)

        ground_truth[metric] = np.mean(metric_bundles)

font = {'size' : font_size}
matplotlib.rc('font', **font)

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16,4))
#fig.suptitle('%s, %s' % (subject,bundle))

for loc,metric in zip(coords, metrics):
    ax[ loc ].hist(data['fixel-'+metric], bins=bins, color=color['fixel-'+metric], label='MRDS')
    if args.dti:
        ax[ loc ].hist(data[metric], bins=bins, color=color[metric], label='DTI')
    if args.ground_truth:
        ax[ loc ].vlines(ground_truth[metric], color=ground_truth_color, ymin=0, ymax=500, linewidth=3)
    ax[ loc ].set_xlabel(metric)
    #ax[ loc ].set_facecolor(background_color)
    ax[ loc ].legend(loc='upper right')
    ax[ loc ].grid(True)
    ax[ loc ].set_axisbelow(True)
plt.show()
