import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import json, os, itertools, sys, matplotlib, argparse
from utils import phantom_info

parser = argparse.ArgumentParser(description='Plot output tractometry statistics')
parser.add_argument('input_path', help='input path with the results of the tractometry')
parser.add_argument('--bundles', nargs='*', help='bundles. [all]')
parser.add_argument('--metrics', default=['FA','RD','AD'], nargs='*', help='metrics to the included. [FA,RD,AD]')
parser.add_argument('--subject', default='group', help='subject. [group]')
parser.add_argument('--phantom', default='templates/Phantomas', help='phantom template. [templates/Phantomas]')
parser.add_argument('--ground_truth', help='path to the ground truth')
parser.add_argument('-nufo', help='add NuFO to the plot', action='store_true')
parser.add_argument('-second_var', help='add second variance to the plot', action='store_true')
args = parser.parse_args()

phantom = args.phantom
nbundles = phantom_info[phantom]['nbundles']

results_tractometry = args.input_path
if args.bundles:
    bundles = ['bundle-%d'%int(i) for i in args.bundles]
else:
    bundles = ['bundle-%d'%(i+1) for i in range(nbundles)]
metrics = args.metrics
subject = args.subject
ground_truth_path = args.ground_truth
nufo = args.nufo

npoints = 20

"""loc = {
    'FA': 0,
    'MD': 1,
    'RD': 2,
    'AD': 3,
    'fixel-FA': 0,
    'fixel-MD': 1,
    'fixel-RD': 2,
    'fixel-AD': 3
}"""
loc = {
    'FA': 0,
    'RD': 1,
    'AD': 2,
    'fixel-FA': 0,
    'fixel-RD': 1,
    'fixel-AD': 2
}


#background_color = '#202946'
#background_color = '#D1D2D3'
#background_color = '#292C35'
#background_color = '#FFFFFF'
background_color = '#E5E5E5'

#ground_truth_color = '#000000'
ground_truth_color = '#D92230'
nufo_color = 'black' #'#FF7F00'

color = {
    'FA': '#1f77b4',
    'MD': '#1f77b4',
    'RD': '#1f77b4',
    'AD': '#1f77b4',
    'fixel-FA': '#2ca02c',
    'fixel-MD': '#2ca02c',
    'fixel-RD': '#2ca02c',
    'fixel-AD': '#2ca02c'
}

facecolor = {
    'FA': '#aec7e8',
    'MD': '#aec7e8',
    'RD': '#aec7e8',
    'AD': '#aec7e8',
    'fixel-FA': '#98df8a',
    'fixel-MD': '#98df8a',
    'fixel-RD': '#98df8a',
    'fixel-AD': '#98df8a'
}

yticks = {
    'FA': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    'MD': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'RD': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'AD': [0.4, 0.8, 1.2, 1.6, 2.0, 2.4]
}

def is_valid(bundle_name, metric):
    dti_metrics = ['ad_metric', 'rd_metric', 'md_metric', 'fa_metric']

    if (bundle_name in metric) or (metric in dti_metrics):
        return True
    
    return False

def rename_metric(bundle_name, metric):
    if 'ad' in metric:
        if bundle_name in metric:
            return 'fixel-AD'
        else:
            return 'AD'
    elif 'rd' in metric:
        if bundle_name in metric:
            return 'fixel-RD'
        else:
            return 'RD'
    elif 'md' in metric:
        if bundle_name in metric:
            return 'fixel-MD'
        else:
            return 'MD'
    elif 'fa' in metric:
        if bundle_name in metric:
            return 'fixel-FA'
        else:
            return 'FA'
    elif 'ihMTR' in metric:
        return 'ihMTR'
    elif 'ihMTsat' in metric:
        return 'ihMTsat'
    elif 'MTR' in metric:
        return 'MTR'
    elif 'MTsat' in metric:
        return 'MTsat'
    elif 'afd' in metric:
        return 'fixel-AFD'
    elif 'nufo' in metric:
        return 'NuFO'

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

def plot_subject_stats(means, stds, alpha=0.95, font_size=55, linewith=7, gt=None):
    font = {'size' : font_size}
    matplotlib.rc('font', **font)
    plt.rcParams["font.family"] = "Times New Roman"

    for bundle in bundles:
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(50,8))
        #fig.suptitle('%s, %s' % (subject,bundle))

        for metric in metrics:
            mean = np.array(means[(bundle,metric)]).ravel()
            std  = np.array(stds[(bundle,metric)]).ravel()
            fixel_mean = np.array(means[(bundle,'fixel-'+metric)]).ravel()
            fixel_std  = np.array(stds[(bundle,'fixel-'+metric)]).ravel()
            dim = np.arange(1, len(mean)+1, 1)

            if metric!='FA':
                mean *= 1e3
                std *= 1e3
                fixel_mean *= 1e3
                fixel_std *= 1e3
                if ground_truth != None:
                    gt[(bundle,metric)] *= 1e3

            # plot ground truth
            if ground_truth != None:
                ax[ loc[metric] ].plot(dim, gt[(bundle,metric)], color=ground_truth_color, linewidth=7, solid_capstyle='round', label='GT', zorder=35)

            # plot DTI metrics
            if args.second_var:
                ax[ loc[metric] ].fill_between(dim, mean-2*std, mean+2*std, facecolor=lighten_color(color[metric], 1.25), edgecolor='black', zorder=0)
            ax[ loc[metric] ].fill_between(dim, mean-std, mean+std, facecolor=facecolor[metric], zorder=5)
            ax[ loc[metric] ].plot(dim, mean, color=color[metric], linewidth=linewith, solid_capstyle='round', label=metric, zorder=10)

            # plot MTM metrics
            if args.second_var:
                ax[ loc[metric] ].fill_between(dim, fixel_mean-2*fixel_std, fixel_mean+2*fixel_std, facecolor=lighten_color(color['fixel-'+metric], 1.25), edgecolor='black', zorder=15)
            ax[ loc[metric] ].fill_between(dim, fixel_mean-fixel_std, fixel_mean+fixel_std, facecolor=facecolor['fixel-'+metric], zorder=20)
            ax[ loc[metric] ].plot(dim, fixel_mean, color=color['fixel-'+metric], linewidth=linewith, solid_capstyle='round', label='fixel-'+metric, zorder=25)

            # add plot info
            #ax[ loc[metric] ].set_xlabel('Location along the Bundle')
            if metric != 'FA':
                ax[ loc[metric] ].set_ylabel(metric + r' [$\mu m^2/ms$]')
            else:
                ax[ loc[metric] ].set_ylabel(metric)
            ax[ loc[metric] ].set_facecolor(background_color)
            ax[ loc[metric] ].set_xticks( np.arange(1, len(mean)+1, 2) )
            ax[ loc[metric] ].set_yticks( yticks[metric] )
            #ax[ loc[metric] ].legend(loc='upper right')
            #ax[ loc[metric] ].grid(True)

            if args.nufo:
                ax2 = ax[ loc[metric] ].twinx()
                ax2.plot(dim,means[(bundle,'NuFO')], linewidth=7, color=nufo_color, solid_capstyle='round', label='N', zorder=30)
                ax2.set_ylim( [0,5] )
                ax2.set_yticks( [0,1,2,3,4,5] )
                #ax2.set_yticks( np.arange(0,120,20) )
                if metric == 'AD':
                    ax2.set_ylabel(r'Ground Truth $N$')
                #if args.legend:
                #ax2.legend(loc='upper right', prop={'size': 12})

        plt.subplots_adjust(wspace=0.25)
        fig.savefig('%s/%s__%s.png' % (results_tractometry,subject,bundle), bbox_inches='tight')
        plt.show()

def load_subject_stats(json_filename):
    json_file = open( json_filename )

    data = json.load( json_file )

    stats_mean = {}
    stats_std = {}

    data = data[subject]

    for bundle in data:
        for metric in data[bundle]:
            if (bundle in metric) or (not 'bundle' in metric): #bundle
                means = []
                stds  = []

                for point in data[bundle][metric]:
                    means.append( data[bundle][metric][point]['mean'] )
                    stds.append( data[bundle][metric][point]['std'] )

                metric = rename_metric(bundle, metric)

                stats_mean[(bundle,metric)] = np.array(means)
                stats_std[(bundle,metric)] = np.array(stds)

    return stats_mean, stats_std

def load_group_stats(json_filename):
    with open(json_filename, 'r+') as f:
        mean_std_per_point = json.load(f)

    stats_means = {}
    stats_stds = {}

    for bundle_name, bundle_stats in mean_std_per_point.items():
        for metric, metric_stats in bundle_stats.items():
            if is_valid(bundle_name, metric):
                nb_points = len(metric_stats)
                num_digits_labels = len(list(metric_stats.keys())[0])
                means = []
                stds = []
                for label_int in range(1, nb_points+1):
                    label = str(label_int).zfill(num_digits_labels)
                    mean = metric_stats.get(label, {'mean': 0})['mean']
                    std = metric_stats.get(label, {'std': 0})['std']
                    if not isinstance(mean, list):
                        mean = [mean]
                        std = [std]

                    means += [mean]
                    stds += [std]

                metric = rename_metric(bundle_name, metric)

                # Robustify for missing data
                means = np.array(list(itertools.zip_longest(*means,
                                                            fillvalue=np.nan))).T
                stds = np.array(list(itertools.zip_longest(*stds,
                                                            fillvalue=np.nan))).T
                for i in range(len(means)):
                    _nan = np.isnan(means[i, :])
                    if np.count_nonzero(_nan) > 0:
                        if np.count_nonzero(_nan) < len(means[i, :]):
                            means[i, _nan] = np.average(means[i, ~_nan])
                            stds[i, _nan] = np.average(stds[i, ~_nan])
                        else:
                            means[i, _nan] = -1
                            stds[i, _nan] = -1

                means = np.average(means, axis=1)
                stds = np.average(stds, axis=1)

                stats_means[(bundle_name, metric)] = means
                stats_stds[(bundle_name, metric)] = stds

    return stats_means, stats_stds

def plot_group_stats(ax, means, stds,  title='', xlabel='', ylabel='', color='#000000', alpha=0.55):
    mean = np.average(means, axis=1)
    std = np.std(means, axis=1)

    dim = np.arange(1, len(mean)+1, 1)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel in ['AD', 'RD', 'MD', 'fixel-AD', 'fixel-RD', 'fixel-MD']:
        ax.set_ylabel(ylabel + r' [$\mu m^2/ms$]')
        mean *= 1e3
        std  *= 1e3
    else:
        ax.set_ylabel(ylabel)
    ax.set_xticks(dim)
    ax.set_facecolor('#E5E5E5')

    ax.plot(dim, mean, linewidth=5, color=color, solid_capstyle='round')
    ax.fill_between(dim, mean-std, mean+std, facecolor=color, alpha=alpha, label=ylabel)

def plot_subject_label_stats():
    X,Y,Z = 16,16,5
    voxels = list(itertools.product(np.arange(X), np.arange(Y), np.arange(Z)))

    label_stats = {}

    for bundle in ['bundle-1', 'bundle-2', 'bundle-3']:
        labels = nib.load( 'full_damage/results_tractometry/sub-001_ses-1/Bundle_Label_And_Distance_Maps/sub-001_ses-1__%s_labels.nii.gz' % bundle ).get_fdata()

        for metric in ['ad', 'rd', 'fa', 'md']:
            stats = [ [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[] ]

            data = nib.load( 'full_damage/results_tractometry/sub-001_ses-1/Fixel_MRDS_Along_Bundle/%s_ic_%s_metric.nii.gz' % (bundle,metric) ).get_fdata()

            for (x,y,z) in voxels:
                label = int(labels[x,y,z])
                stats[label-1].append( data[x,y,z] )

            label_stats[(bundle,metric)] = stats

    font = {'size' : 12}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(20,10))

    for i,bundle in enumerate(['bundle-1', 'bundle-2', 'bundle-3']):
        for j,metric in enumerate(['ad', 'rd', 'fa', 'md']):
            print((bundle,metric))

            stats = label_stats[(bundle,metric)]

            for label in range(len(stats)):
                npoints = len(stats[label])

                color='blue'
                if bundle == 'bundle-2':
                    color = 'red'

                ax[i,j].set_xlabel('Location along the streamline')
                if metric in ['ad', 'rd', 'md']:
                    ax[i,j].set_ylabel(metric + r' [$\mu m^2/ms$]')
                    stats[label] = np.array(stats[label])*1e3
                else:
                    ax[i,j].set_ylabel(metric)

                ax[i,j].set_ylim( ylims[metric.upper()] )
                ax[i,j].hlines(y=gt[(bundle,metric.upper())], xmin=1, xmax=len(stats), linewidth=5, color='red')
                ax[i,j].scatter( [label+1]*npoints, stats[label], color=ms_fixel_color, alpha=0.5 )

    fig.savefig(os.path.join(output_path, 'labels.png'.format(bundle, metric)), bbox_inches='tight')
    #plt.show()

def plot_metrics_stats(means, stds, title, xlabel, ylabel, fill_color):
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if means.ndim > 1:
        mean = np.average(means, axis=1)
        std = np.std(means, axis=1)
        alpha = 0.95
    else:
        mean = np.array(means).ravel()
        std = np.array(stds).ravel()
        alpha = 0.9

    dim = np.arange(1, len(mean)+1, 1)

    ax.plot(dim, mean, color="k", linewidth=5, solid_capstyle='round')
    ax.set_xticks(dim)

    plt.fill_between(dim, mean-std, mean+std, facecolor=fill_color, alpha=alpha)

    plt.close(fig)
    return fig

def plot_metrics_stats_comparison(means, stds, fixel_means, fixel_stds, title, xlabel, ylabel, fill_color):

    color = '#7FBF7F'
    fixel_color = '#7E7EFD'

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        if ylabel in ['AD', 'RD', 'MD']:
            ax.set_ylabel(ylabel + r' [$\mu m^2/ms$]')
        else:
            ax.set_ylabel(ylabel)

    if ylabel in ['AD', 'RD', 'MD']:
        means *= 1e3
        stds *= 1e3
        fixel_means *= 1e3
        fixel_stds *= 1e3

    if means.ndim > 1:
        mean = np.average(means, axis=1)
        fixel_mean = np.average(fixel_means, axis=1)
        std = np.std(means, axis=1)
        fixel_std = np.std(fixel_means, axis=1)
        alpha = 0.55
    else:
        mean = np.array(means).ravel()
        fixel_mean = np.array(fixel_means).ravel()
        std = np.array(stds).ravel()
        fixel_std = np.array(fixel_stds).ravel()
        alpha = 0.9

    dim = np.arange(1, len(mean)+1, 1)

    plt.fill_between(dim, mean-std, mean+std, facecolor=color, alpha=alpha, label=ylabel, edgecolor='black')
    plt.fill_between(dim, fixel_mean-fixel_std, fixel_mean+fixel_std, facecolor=fixel_color, alpha=alpha, label='fixel-'+ylabel, edgecolor='black')

    ax.plot(dim, mean, color=color, linewidth=5, solid_capstyle='round')
    ax.plot(dim, fixel_mean, color=fixel_color, linewidth=5, solid_capstyle='round')
    ax.set_xticks(dim)
    ax.set_facecolor('#E5E5E5')

    plt.legend(loc='upper right', prop={'size': 15})
    plt.close(fig)
    return fig

def load_subject_ground_truth( ground_truth_path ):
    phantom = 'templates/Phantomas'
    nbundles = phantom_info[phantom]['nbundles']
    bundles = ['bundle-%d' % (i+1) for i in range(nbundles)]

    stats_gt = {}

    for bundle in bundles:
        labels = nib.load('%s/%s/Bundle_Label_And_Distance_Maps/%s__%s_labels.nii.gz' % (results_tractometry,subject,subject,bundle)).get_fdata().astype(np.uint8).flatten()
        
        for metric in metrics:
            data = nib.load( '%s/%s/%s__%s.nii.gz'%(ground_truth_path,subject,bundle,metric.lower()) ).get_fdata().flatten()

            mem = []
            gt = np.zeros(npoints)

            for label,val in zip(labels,data):
                if (label > 0) and (val > 0):# and (label not in mem):
                    gt[label-1] = val
                    mem.append(label)

            stats_gt[(bundle,metric)] = gt

    return stats_gt

if subject == 'group':
    means, stds = load_group_stats( '%s/Statistics/mean_std_per_point.json'%(results_tractometry) )
else:
    means, stds = load_subject_stats( '%s/%s/Bundle_Mean_Std_Per_Point/%s__mean_std_per_point.json'%(results_tractometry,subject,subject) )

if args.ground_truth:
    ground_truth = load_subject_ground_truth( ground_truth_path )
else:
    ground_truth = None

plot_subject_stats(means, stds, gt=ground_truth)
