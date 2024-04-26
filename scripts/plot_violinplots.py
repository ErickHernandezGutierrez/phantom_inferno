import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import argparse,matplotlib
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
#metrics = ['FA']

background_color = '#E5E5E5'
ground_truth_color = '#D92230'
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
    'FA': [0.2, 0.4, 0.6, 0.8, 1.0],
    'MD': [0.4, 0.6, 0.8, 1.0],
    'RD': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'AD': [0.4, 0.8, 1.2, 1.6, 2.0, 2.4]
}
ylims = {
    'FA': [0.3, 1.0],
    'MD': [0.5, 0.9],
    'RD': [0.05, 0.6],
    'AD': [0.6, 2.4]
}
ylabels = {
    'FA': 'FA',
    'MD': r'MD [$\mu m^2/ms$]',
    'RD': r'RD [$\mu m^2/ms$]',
    'AD': r'AD [$\mu m^2/ms$]'
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
    'MD': '#9CD1B6',
    'RD': '#9CD1B6',
    'AD': '#9CD1B6',
    'fixel-FA': '#F6AA92',
    'fixel-MD': '#F6AA92',
    'fixel-RD': '#F6AA92',
    'fixel-AD': '#F6AA92'
}

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

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

font_size = 40
font = {'size' : font_size}
matplotlib.rc('font', **font)
plt.rcParams["font.family"] = "Times New Roman"
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(50, 35))

for metric in metrics:
    mrds_data = []
    dti_data  = []

    for i in range(nbundles):
        bundle = 'bundle-%d' % (i+1)

        # load MRDS data
        data = nib.load('%s/%s/Fixel_MRDS/%s_ic_%s_metric.nii.gz'%(results_tractometry,subject,bundle,metric.lower())).get_fdata().flatten()
        data = data[data > 0]
        if metric in ['MD','AD','RD']:
            data *= 1e3
        mrds_data.append( data )

        # load DTI data
        data = (nib.load('%s/results_DTInolin_%s.nii.gz'%(dti_results,metric)).get_fdata() * masks[:,:,:, i]).flatten()
        data = data[data > 0]
        if metric in ['MD','AD','RD']:
            data *= 1e3
        dti_data.append(data)

    # plot data
    violin = ax[ loc[metric] ].violinplot(dti_data, showmeans=False, showmedians=True, showextrema=False, positions=[i+1 for i in range(nbundles)])
    for pc in violin['bodies']:
        pc.set_facecolor( lighten_color(color[metric], 0.2) )
        pc.set_edgecolor( color[metric] )
        pc.set_linewidth(3)
        pc.set_alpha(0.75)
    for partname in ['cmedians']:
        vp = violin[partname]
        vp.set_edgecolor( color[metric] )
        vp.set_linewidth(3)

    violin = ax[ loc[metric] ].violinplot(mrds_data, showmeans=False, showmedians=True, showextrema=False)
    for pc in violin['bodies']:
        pc.set_facecolor( lighten_color(color['fixel-'+metric], 0.2) )
        pc.set_edgecolor( color['fixel-'+metric] )
        pc.set_linewidth(3)
        pc.set_alpha(0.75)
    for partname in ['cmedians']:
        vp = violin[partname]
        vp.set_edgecolor( color['fixel-'+metric] )
        vp.set_linewidth(3)

    ax[ loc[metric] ].hlines(y=ground_truth[metric], xmin=0.85, xmax=20.15, color=ground_truth_color, linewidth=3, zorder=0)
    ax[ loc[metric] ].set_xticks([i+1 for i in range(nbundles)])
    ax[ loc[metric] ].set_yticks(yticks[metric])
    ax[ loc[metric] ].set_ylabel(ylabels[metric])
    ax[ loc[metric] ].set_ylim(ylims[metric])
    ax[ loc[metric] ].set_xlabel('Bundle')
    ax[ loc[metric] ].set_facecolor(background_color)

fig.savefig('%s/violinplots.png' % (results_tractometry), bbox_inches='tight')
plt.show()
