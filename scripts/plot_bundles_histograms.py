import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
import argparse,matplotlib,joypy
from utils import phantom_info

parser = argparse.ArgumentParser(description='Plot output tractometry statistics')
parser.add_argument('input_path', help='input path with the results of the tractometry')
parser.add_argument('--bundles', nargs='*', help='bundles. [all]')
parser.add_argument('--subject', default='group', help='subject. [group]')
parser.add_argument('--phantom', default='templates/Phantomas', help='phantom template. [templates/Phantomas]')
parser.add_argument('--ground_truth', help='path to the ground truth')
parser.add_argument('--dti', help='path to the dti results')
args = parser.parse_args()

phantom = args.phantom
nbundles = phantom_info[phantom]['nbundles']
X,Y,Z = phantom_info[phantom]['dims']
results_tractometry = args.input_path
if args.bundles:
    bundles = ['bundle-%d'%int(i) for i in args.bundles]
else:
    bundles = ['bundle-%d'%(i+1) for i in range(nbundles)]
subject = args.subject
results_ground_truth = args.ground_truth
dti_results = args.dti

metrics = ['FA', 'MD', 'RD', 'AD']
metrics = ['FA']
loc = {
    'FA': 0,
    'MD': 1,
    'RD': 2,
    'AD': 3
}
ground_truth = {
    'FA': [],
    'MD': [],
    'AD': [],
    'RD': []
}
yticks = {
    'FA': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    'MD': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    'RD': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'AD': [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4]
}
ylabels = {
    'FA': 'FA',
    'MD': r'MD [$\mu m^2/ms$]',
    'RD': r'RD [$\mu m^2/ms$]',
    'AD': r'AD [$\mu m^2/ms$]'
}
edgecolor = {
    'FA': '#0E7D48',
    'MD': '#0E7D48',
    'RD': '#0E7D48',
    'AD': '#0E7D48',
    'fixel-FA': '#AD151D',
    'fixel-MD': '#AD151D',
    'fixel-RD': '#AD151D',
    'fixel-AD': '#AD151D'
}
facecolor = {
    'FA': '#9CD1B6',
    'MD': '#9CD1B6',
    'RD': '#9CD1B6',
    'AD': '#9CD1B6',
    'fixel-FA': '#F6AA92',
    'fixel-MD': '#F6AA92',
    'fixel-RD': '#F6AA92',
    'fixel-AD': '#F6AA92'
}

# load WM masks
masks = np.zeros((X,Y,Z,nbundles), dtype=np.uint8)
for i in range(nbundles):
    mask_filename = '%s/bundle-%d__wm-mask.nii.gz' % (phantom,i+1)
    masks[:,:,:, i] = nib.load( mask_filename ).get_fdata().astype(np.uint8)

if args.ground_truth:
    for metric in metrics:
        metric_bundles = np.array([])
        for i in range(nbundles):
            metric_bundle = nib.load( '%s/bundle-%d__%s.nii.gz'%(results_ground_truth,i+1,metric.lower()) ).get_fdata() * masks[:,:,:, i]

            if metric in ['MD','AD','RD']:
                metric_bundle *= 1e3

            metric_bundle = metric_bundle[metric_bundle > 0]
            metric_bundles = np.concatenate((metric_bundles, metric_bundle), axis=None)
        ground_truth[metric] = np.mean(metric_bundles)

font_size = 25
font = {'size' : font_size}
matplotlib.rc('font', **font)
plt.rcParams["font.family"] = "Times New Roman"
#fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(50, 15), sharex=True, sharey=True)

for metric in metrics:
    mrds_data = np.array([])
    dti_data  = np.array([])
    bundle_data = np.array([])

    for i in range(nbundles):
        bundle = 'bundle-%d' % (i+1)

        # load MRDS data
        data = nib.load('%s/%s/Fixel_MRDS/%s_ic_%s_metric.nii.gz'%(results_tractometry,subject,bundle,metric.lower())).get_fdata() * masks[:,:,:, i]
        data = data - (1 - masks[:,:,:, i])
        data = data.flatten()
        data = data[data >= 0]
        if metric in ['MD','AD','RD']:
            data *= 1e3
        bundle_data = np.concatenate((bundle_data,np.array(['bundle-%d'%(i+1) for x in range(data.shape[0])])), axis=None)
        mrds_data = np.concatenate((mrds_data,data),axis=None)
        #mrds_data.append( data )
        #print(data.shape)

        # load DTI data
        data = (nib.load('%s/results_DTInolin_%s.nii.gz'%(dti_results,metric)).get_fdata() * masks[:,:,:, i])
        data = data - (1 - masks[:,:,:, i])
        data = data.flatten()
        data = data[data >= 0]
        if metric in ['MD','AD','RD']:
            data *= 1e3
        dti_data = np.concatenate((dti_data,data),axis=None)
        #dti_data.append( data )
        #print(data.shape)

        #ax[0,i].hist(mrds_data[i], bins=32, color='green', alpha=0.5)
        #ax[0,i].hist(dti_data[i], bins=32, color='blue', alpha=0.5)

    if args.ground_truth:
        metric_bundles = np.array([])
        for i in range(nbundles):
            metric_bundle = nib.load( '%s/bundle-%d__%s.nii.gz'%(results_ground_truth,i+1,metric.lower()) ).get_fdata() * masks[:,:,:, i]

            if metric in ['MD','AD','RD']:
                metric_bundle *= 1e3

            metric_bundle = metric_bundle[metric_bundle > 0]
            metric_bundles = np.concatenate((metric_bundles, metric_bundle), axis=None)

        ground_truth[metric] = np.mean(metric_bundles)

print(mrds_data.shape)
print(dti_data.shape)
print(bundle_data.shape)

df = pd.DataFrame({'MRDS': mrds_data,
                   'DTI': dti_data,
                   'group': bundle_data})

fig, axes = joypy.joyplot(df, by = "group")#, hist=True, bins=32)
for ax in axes:
    ax.set_xlim([0.4, 1])
    ax.vlines(ground_truth['FA'], color='black', ymin=0, ymax=31000, linewidth=3, label='GT')

"""
    # plot data
    violin = ax[ loc[metric] ].violinplot(mrds_data, showmeans=False, showmedians=True, showextrema=False)
    for pc in violin['bodies']:
        pc.set_facecolor( facecolor['fixel-'+metric] )
        pc.set_edgecolor( edgecolor['fixel-'+metric] )
        pc.set_alpha(1.0)
    for partname in ['cmedians']:
        vp = violin[partname]
        vp.set_edgecolor( edgecolor['fixel-'+metric] )
        vp.set_linewidth(5)

    violin = ax[ loc[metric] ].violinplot(dti_data, showmeans=False, showmedians=True, showextrema=False, positions=[i+1.15 for i in range(nbundles)])
    for pc in violin['bodies']:
        pc.set_facecolor( facecolor[metric] )
        pc.set_edgecolor( edgecolor[metric] )
        pc.set_alpha(1.0)
    for partname in ['cmedians']:
        vp = violin[partname]
        vp.set_edgecolor( edgecolor[metric] )
        vp.set_linewidth(5)

    ax[ loc[metric] ].hlines(y=ground_truth[metric], xmin=0.85, xmax=20.15, color='black', linewidth=3)
    ax[ loc[metric] ].set_xticks([i+1 for i in range(nbundles)])
    ax[ loc[metric] ].set_yticks(yticks[metric])
    ax[ loc[metric] ].set_ylabel(ylabels[metric])
    ax[ loc[metric] ].set_xlabel('Bundle')
"""

fig.savefig('%s/histograms.png' % (results_tractometry), bbox_inches='tight')
plt.show()
