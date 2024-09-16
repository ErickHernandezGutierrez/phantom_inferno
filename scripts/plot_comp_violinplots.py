import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import argparse,matplotlib
from utils import phantom_info

parser = argparse.ArgumentParser(description='Plot output tractometry statistics')
parser.add_argument('--input_path', default='./', help='input path with the results of the tractometry')
parser.add_argument('--bundles', nargs='*', help='bundles. [all]')
parser.add_argument('--subject', default='sub-001_ses-1', help='subject. [group]')
parser.add_argument('--phantom', default='templates/Phantomas', help='phantom template. [templates/Phantomas]')
parser.add_argument('--ground_truth', help='path to the ground truth')
parser.add_argument('--dti', help='path to the dti results')
parser.add_argument('-lesion', help='add lesion', action='store_true')
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

#metrics = ['FA', 'MD', 'RD', 'AD']
metrics = ['FA', 'RD', 'AD']

background_color = '#E5E5E5'
ground_truth_color = '#D92230'
loc = {
    'FA': 0,
    'RD': 1,
    'AD': 2
}
"""loc = {
    'FA': (0,0),
    'MD': (0,1),
    'RD': (1,0),
    'AD': (1,1)
}"""

ground_truth = {}
ground_truth_lesion = {}

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

# load lesion masks
lesion_masks = np.zeros((X,Y,Z,nbundles), dtype=np.uint8)
for i in range(nbundles):
    mask_filename = '%s/bundle-%d__lesion-mask.nii.gz' % (phantom,i+1)
    lesion_masks[:,:,:, i] = nib.load( mask_filename ).get_fdata().astype(np.uint8)

font_size = 55
font = {'size' : font_size}
matplotlib.rc('font', **font)
plt.rcParams["font.family"] = "Times New Roman"
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(50, 15))

lesion_bundles = [12]#[4, 8, 12, 19]
scenarios = ['control','bigdemyelination','axonloss']

all_mrds_data = {}
all_dti_data = {}

for scenario in scenarios:
    for metric in metrics:
        mrds_data = []
        dti_data  = []

        for i in lesion_bundles:
            bundle = 'bundle-%d' % (i+1)

            # load MRDS data
            data = nib.load('experiments/%s/results_tractometry/%s/Fixel_MRDS/%s_ic_%s_metric.nii.gz'%(scenario,subject,bundle,metric.lower())).get_fdata().flatten()
            data = data[data > 0]
            if metric in ['MD','AD','RD']:
                data *= 1e3
            mrds_data.append(data)

            # load DTI data
            data = (nib.load('experiments/%s/%s/dti/results_DTInolin_%s.nii.gz'%(scenario,subject,metric)).get_fdata() * masks[:,:,:, i]).flatten()
            data = data[data > 0]
            if metric in ['MD','AD','RD']:
                data *= 1e3
            dti_data.append(data)

            # load GT data
            gt_data = nib.load( 'experiments/%s/ground_truth/%s/bundle-%d__%s.nii.gz'%(scenario,subject,i+1,metric.lower()) ).get_fdata()

            if metric in ['MD','AD','RD']:
                gt_data *= 1e3

            control = gt_data * (masks[:,:,:, i]-lesion_masks[:,:,:, i])
            lesion  = gt_data * lesion_masks[:,:,:, i]

            control = control[control > 0]
            lesion = lesion[lesion > 0]
            
            print((scenario,metric))
            ground_truth[(scenario,metric)] = np.mean(control)
            ground_truth_lesion[(scenario,metric)] = np.mean(lesion)

        all_mrds_data[(scenario,metric)] = mrds_data
        all_dti_data[(scenario,metric)] = dti_data

def set_violin(violin, color, body_linewidth=1, part_linewidth=5, alpha=0.75):
    for pc in violin['bodies']:
        pc.set_facecolor( color )
        pc.set_edgecolor( lighten_color(color, 1.5) )
        pc.set_linewidth( body_linewidth )
        pc.set_alpha( alpha )
    for partname in ['cmeans']:#,'cmedians'):
        vp = violin[partname]
        vp.set_edgecolor( lighten_color(color, 1.2) )
        vp.set_linewidth( part_linewidth )

pos = [0,2,4,6]
pos = [0]

for metric in metrics:
    violin = ax[ loc[metric] ].violinplot(all_dti_data[('control',metric)], showmeans=True, showmedians=False, showextrema=False, positions=[i+1-0.5 for i in pos])
    set_violin(violin, color[metric])

    violin = ax[ loc[metric] ].violinplot(all_mrds_data[('control',metric)], showmeans=True, showmedians=False, showextrema=False, positions=[i+1-0.5 for i in pos])
    set_violin(violin, color['fixel-'+metric])

    violin = ax[ loc[metric] ].violinplot(all_dti_data[('bigdemyelination',metric)], showmeans=True, showmedians=False, showextrema=False, positions=[i+1 for i in pos])
    set_violin(violin, color[metric])
    violin = ax[ loc[metric] ].violinplot(all_mrds_data[('bigdemyelination',metric)], showmeans=True, showmedians=False, showextrema=False, positions=[i+1 for i in pos])
    set_violin(violin, color['fixel-'+metric])

    violin = ax[ loc[metric] ].violinplot(all_dti_data[('axonloss',metric)], showmeans=True, showmedians=False, showextrema=False, positions=[i+1+0.5 for i in pos])
    set_violin(violin, color[metric])
    violin = ax[ loc[metric] ].violinplot(all_mrds_data[('axonloss',metric)], showmeans=True, showmedians=False, showextrema=False, positions=[i+1+0.5 for i in pos])
    set_violin(violin, color['fixel-'+metric])

    # plot ground-truth
    ax[ loc[metric] ].hlines(y=ground_truth[('control',metric)], xmin=0.35, xmax=0.65, color=ground_truth_color, linewidth=5, zorder=10)

    ax[ loc[metric] ].hlines(y=ground_truth[('bigdemyelination',metric)], xmin=0.85, xmax=1.15, color=ground_truth_color, linewidth=5, zorder=10)
    ax[ loc[metric] ].hlines(y=ground_truth_lesion[('bigdemyelination',metric)], xmin=0.85, xmax=1.15, color='purple', linewidth=5, zorder=10)

    ax[ loc[metric] ].hlines(y=ground_truth[('axonloss',metric)], xmin=1.35, xmax=1.65, color=ground_truth_color, linewidth=5, zorder=10)
    ax[ loc[metric] ].hlines(y=ground_truth_lesion[('axonloss',metric)], xmin=1.35, xmax=1.65, color='purple', linewidth=5, zorder=10)

    #ax[ loc[metric] ].set_xticks([1,3,5,7], labels=np.array(['bundle %d'%(i+1) for i in lesion_bundles]))
    ax[ loc[metric] ].set_xticks([1], labels=np.array(['bundle %d'%(i+1) for i in lesion_bundles]))

    ax[ loc[metric] ].set_yticks(yticks[metric])
    ax[ loc[metric] ].set_ylabel(ylabels[metric])
    #ax[ loc[metric] ].set_ylim(ylims[metric])
    ax[ loc[metric] ].set_facecolor(background_color)

fig.savefig('comp_violinplots.png', bbox_inches='tight')
plt.show()
