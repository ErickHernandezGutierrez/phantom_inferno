#!/usr/bin/python
import os,argparse,matplotlib
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Plot MRDS estimated metrics histograms')
parser.add_argument('results_mrds', help='path to the MRDS results')
parser.add_argument('--prefix', default='results', help='MRDS prefix. [results]')
parser.add_argument('--modsel', default='bic', help='MRDS model selector (aic,bic,ftest). [bic]')
parser.add_argument('--mask', help='optional mask filename')
parser.add_argument('--bins', default=16, type=int, help='number of bins of the histograms. [64]')
parser.add_argument('--dti', help='path to the DTI results')
parser.add_argument('--ground_truth', help='path to the ground truth')
args = parser.parse_args()

results_mrds = args.results_mrds
results_dti = args.dti
results_gt = args.ground_truth
prefix = args.prefix
modsel = args.modsel
mask_filename = args.mask
bins = args.bins

metrics = ['FA','MD','AD','RD']
#metrics = ['fixel-FA','fixel-MD','fixel-AD','fixel-RD']
#coords  = [0,1,2,3]
coords  = [(0,0),(0,1),(1,0),(1,1)]
background_color = '#E5E5E5'
ground_truth_color = '#D92230'

legend_loc = {
    'FA': 'upper left',
    'MD': 'upper right',
    'RD': 'upper right',
    'AD': 'upper left',
}

color = {
    'FA': '#4475B1',
    'MD': '#4475B1',
    'RD': '#4475B1',
    'AD': '#4475B1',
    'fixel-FA': '#4DAE4A',
    'fixel-MD': '#4DAE4A',
    'fixel-RD': '#4DAE4A',
    'fixel-AD': '#4DAE4A'
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
xticks = {
    'FA': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    'MD': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'RD': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'AD': [0.4, 0.8, 1.2, 1.6, 2.0, 2.4]
}
xlabels = {
    'FA': 'FA',
    'MD': r'MD [$\mu m^2/ms$]',
    'RD': r'RD [$\mu m^2/ms$]',
    'AD': r'AD [$\mu m^2/ms$]',
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

"""
for metric in ground_truth:
    if metric != 'FA':
        print('%s: %.16f' % (metric,ground_truth[metric]*1e-3))
    else:
        print('%s: %.16f' % (metric,ground_truth[metric]))
"""

font_size = 30
font = {'size' : font_size}
matplotlib.rc('font', **font)
plt.rcParams["font.family"] = "Times New Roman"

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16,8))
#fig.suptitle('%s, %s' % (subject,bundle))

for loc,metric in zip(coords, metrics):
    # plot MRDS histograms
    if metric in ['AD','RD','MD']:
        z = 0
    else:
        z = 5
    ax[ loc ].hist(data['fixel-'+metric], bins=bins, color=color['fixel-'+metric], label='MRDS', alpha=1, zorder=z)

    # plot DTI histograms
    if args.dti:
        ax[ loc ].hist(data[metric], bins=bins, color=color[metric], label='DTI', alpha=0.5, zorder=5-z)

    # plot GT
    if args.ground_truth:
        ax[ loc ].vlines(ground_truth[metric], color=ground_truth_color, ymin=0, ymax=31000, linewidth=5, label='GT', zorder=10)
    
    ax[ loc ].set_xlabel(xlabels[metric])
    ax[ loc ].set_facecolor(background_color)
    ax[ loc ].legend(loc=legend_loc[metric], fontsize=25)
    ax[ loc ].set_xticks(xticks[metric])
    #ax[ loc ].grid(True)
    ax[ loc ].set_axisbelow(True)

plt.subplots_adjust(wspace=0.25, hspace=0.5)
plt.show()
fig.savefig('%s/histograms.png' % (results_mrds), bbox_inches='tight')
