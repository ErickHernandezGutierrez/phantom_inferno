import numpy as np
import nibabel as nib
import json, os, itertools, sys, matplotlib
import matplotlib.pyplot as plt

#json_filename = sys.argv[1]
#output_path = sys.argv[2]

experiment = sys.argv[1]
subject = sys.argv[2]
output_path = sys.argv[3]

font = {'size' : 20}
matplotlib.rc('font', **font)

ms_color = '#7FBF7F'
ms_fixel_color = '#7E7EFD'

ylims = {
    'AD': [0.5, 2.0],
    'RD': [0.0, 0.6],
    'FA': [0.3, 1.0],
    'MD': [0.5, 0.8]
}

if experiment in ['medium_damage', 'medium_damage_NAWM', 'upsampled_medium_lesion']:
    l = 5
    m = 11
    r = 4
elif experiment == 'full_damage' or experiment=='crossing_full_damage':
    l = 0
    m = 20
    r = 0
elif experiment == 'control':
    l = 20
    m = 0
    r = 0
elif experiment == 'crossing':
    l = 5
    m = 11
    r = 4

gt_ad = nib.load('%s/%s/ground_truth/ad.nii' % (experiment,subject)).get_fdata()
gt_rd = nib.load('%s/%s/ground_truth/rd.nii' % (experiment,subject)).get_fdata()
gt_fa = nib.load('%s/%s/ground_truth/fa.nii' % (experiment,subject)).get_fdata()
gt_md = nib.load('%s/%s/ground_truth/md.nii' % (experiment,subject)).get_fdata()
gt = {
    ('bundle-1','AD'): np.array( [gt_ad[4,8,2, 0]*1e3]*20 ),
    ('bundle-1','RD'): np.array( [gt_rd[4,8,2, 0]*1e3]*20 ),
    ('bundle-1','FA'): np.array( [gt_fa[4,8,2, 0]]    *20 ),
    ('bundle-1','MD'): np.array( [gt_md[4,8,2, 0]*1e3]*20 ),

    ('bundle-2','AD'): np.array( [gt_ad[0,15,2, 1]*1e3]*l + [gt_ad[7,7,2, 1]*1e3]*m + [gt_ad[0,15,2, 1]*1e3]*r ),
    ('bundle-2','RD'): np.array( [gt_rd[0,15,2, 1]*1e3]*l + [gt_rd[7,7,2, 1]*1e3]*m + [gt_rd[0,15,2, 1]*1e3]*r ),
    ('bundle-2','FA'): np.array( [gt_fa[0,15,2, 1]]    *l + [gt_fa[7,7,2, 1]]    *m + [gt_fa[0,15,2, 1]]    *r ),
    ('bundle-2','MD'): np.array( [gt_md[0,15,2, 1]*1e3]*l + [gt_md[7,7,2, 1]*1e3]*m + [gt_md[0,15,2, 1]*1e3]*r ),

    ('bundle-3','AD'): np.array( [gt_ad[4,8,2, 2]*1e3]*14 ),
    ('bundle-3','RD'): np.array( [gt_rd[4,8,2, 2]*1e3]*14 ),
    ('bundle-3','FA'): np.array( [gt_fa[4,8,2, 2]]    *14 ),
    ('bundle-3','MD'): np.array( [gt_md[4,8,2, 2]*1e3]*14 )
}

coords = {
    # bundles
    'bundle-1': 0,
    'bundle-2': 1,
    'bundle-3': 2,

    # metrics
    'AD' : 0,
    'RD' : 1,
    'FA' : 2,
    'MD' : 3,
}

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
        return 'AFD'
    elif 'nufo' in metric:
        return 'NuFO'

def load_subject_stats(json_filename):
    json_file = open( json_filename )

    data = json.load( json_file )

    stats_mean = {}
    stats_std = {}

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

def plot_subject_stats(ax, means, stds, title='', xlabel='', ylabel='', color='#000000', facecolor='#000000', alpha=0.55):
    mean = np.array(means).ravel()
    std = np.array(stds).ravel()

    dim = np.arange(1, len(mean)+1, 1)

    #ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel in ['AD', 'RD', 'MD', 'fixel-AD', 'fixel-RD', 'fixel-MD']:
        ax.set_ylabel(ylabel + r' [$\mu m^2/ms$]')
        mean *= 1e3
        std  *= 1e3
    else:
        ax.set_ylabel(ylabel)
    ax.set_xticks( np.arange(1, len(mean)+1, 5) )
    ax.set_facecolor('#E5E5E5')

    ax.plot(dim, mean, linewidth=5, color=color, solid_capstyle='round')
    ax.fill_between(dim, mean-std, mean+std, facecolor=facecolor, label=ylabel, alpha=0.55)

def load_group_stats(json_filename):
    with open(json_filename, 'r+') as f:
        #mean_std_per_point = json.load(f)
        mean_std_per_point = list(json.load(f).values())[0]

    stats_means = {}
    stats_stds = {}

    for bundle_name, bundle_stats in mean_std_per_point.items():
        for metric, metric_stats in bundle_stats.items():
            if (bundle_name in metric) or (not 'bundle' in metric): #bundle
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

                color = '0x727272'

                metric = rename_metric(bundle_name, metric)

                print( (bundle_name,metric) )

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

                ########
                means = np.squeeze(means)
                stds = np.squeeze(stds)

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

#plot_subject_label_stats()

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

def plot_gt_stats(ax, mean, title='', xlabel='', ylabel='', color='#000000', alpha=0.55):
    dim = np.arange(1, len(mean)+1, 1)

    #if ylabel in ['AD', 'RD', 'MD', 'fixel-AD', 'fixel-RD', 'fixel-MD']:
        #mean *= 1e3
    #ax.set_facecolor('#E5E5E5')

    ax.plot(dim, mean, linewidth=5, color='red', solid_capstyle='round', label='GT '+ylabel)

json_filename = '%s/results_tractometry/Statistics/mean_std_per_point.json' % (experiment)

ms_stats_mean, ms_stats_std = load_subject_stats(json_filename)
#ms_stats_mean, ms_stats_std = load_group_stats(json_filename)

gt_stats_mean = {
    ('bundle-1', 'AD'): np.array([0.002413370] * 20),
    ('bundle-1', 'RD'): np.array([0.000306951] * 20),
    ('bundle-1', 'FA'): np.array([0.859027] * 20),
    ('bundle-1', 'MD'): np.array([0.00100909] * 20),

    ('bundle-2', 'AD'): np.array([0.00245575]*5 + [0.00169477]*11 + [0.00245575]*4),
    ('bundle-2', 'RD'): np.array([0.000249684]*5 + [0.000588761]*11 + [0.000249684]*4),
    ('bundle-2', 'FA'): np.array([0.889182]*5 + [0.585728]*11 + [0.889182]*4),
    ('bundle-2', 'MD'): np.array([0.000985041]*5 + [0.00095743]*11 + [0.000985041]*4),

    ('bundle-3', 'AD'): np.array([0.00258506] * 13),
    ('bundle-3', 'RD'): np.array([0.000295041] * 13),
    ('bundle-3', 'FA'): np.array([0.874548] * 13),
    ('bundle-3', 'MD'): np.array([0.00105838] * 13),
}

#ms_color = '#CF0201' #'#727272'
#ms_fixel_color = '#F9C306'


"""for bundle in ['bundle-1', 'bundle-2', 'bundle-3']:
    for metric in ['AD', 'RD', 'FA', 'MD']:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9,6))

        #plot_group_stats(ax, ms_stats_means[(bundle, 'fixel-'+metric)], ms_stats_stds[(bundle, 'fixel-'+metric)], 'fixel-'+metric, 'Location along the streamline', 'fixel-'+metric, ms_fixel_color)
        #plot_group_stats(ax, ms_stats_means[(bundle, metric)], ms_stats_stds[(bundle, metric)], metric, 'Location along the streamline', metric, ms_color)

        ax.set_ylim( ylims[metric] )

        plot_subject_stats(ax, ms_stats_mean[(bundle, 'fixel-'+metric)], ms_stats_std[(bundle, 'fixel-'+metric)], 'fixel-'+metric,            'Location along the streamline', 'fixel-'+metric, ms_fixel_color, ms_fixel_color)
        plot_subject_stats(ax, ms_stats_mean[(bundle, metric)],          ms_stats_std[(bundle, metric)],          'Tract %s Profiles vs Tract fixel-%s Profiles'%(metric,metric), 'Location along the streamline', metric, ms_color, ms_color)

        #ax.hlines(gt[(bundle,metric)], xmin=1, xmax=len(ms_stats_mean[(bundle, metric)]), linewidth=5, color='red', label='GT %s'%metric)

        plot_gt_stats(ax, gt[(bundle, metric)], ylabel=metric)

        #plt.legend(loc='upper right', prop={'size': 15})
        plt.close(fig)
        fig.savefig(os.path.join(output_path, '{}_{}.png'.format(bundle, metric)), bbox_inches='tight')
#"""

fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(40,25))
for i,bundle in enumerate(['bundle-1', 'bundle-2']):#, 'bundle-3']):
    for j,metric in enumerate(['AD', 'RD', 'FA', 'MD']):

        #plot_group_stats(ax, ms_stats_means[(bundle, 'fixel-'+metric)], ms_stats_stds[(bundle, 'fixel-'+metric)], 'fixel-'+metric, 'Location along the streamline', 'fixel-'+metric, ms_fixel_color)
        #plot_group_stats(ax, ms_stats_means[(bundle, metric)], ms_stats_stds[(bundle, metric)], metric, 'Location along the streamline', metric, ms_color)

        ax[i,j].set_ylim( ylims[metric] )

        plot_subject_stats(ax[i,j], ms_stats_mean[(bundle, 'fixel-'+metric)], ms_stats_std[(bundle, 'fixel-'+metric)], 'fixel-'+metric,            'Location along the streamline', 'fixel-'+metric, ms_fixel_color, ms_fixel_color)
        plot_subject_stats(ax[i,j], ms_stats_mean[(bundle, metric)],          ms_stats_std[(bundle, metric)],          'Tract %s Profiles vs Tract fixel-%s Profiles'%(metric,metric), 'Location along the streamline', metric, ms_color, ms_color)

        #ax[i,j].hlines(gt[(bundle,metric)], xmin=1, xmax=len(ms_stats_mean[(bundle, metric)]), linewidth=5, color='red', label='GT %s'%metric)

        plot_gt_stats(ax[i,j], gt[(bundle, metric)], ylabel=metric)

        #ax[i,j].legend(loc='upper right', prop={'size': 15})

plt.show()
plt.close(fig)
#fig.savefig(os.path.join(output_path, 'calis.png'), bbox_inches='tight')
fig.savefig(os.path.join(output_path, 'tract-profiles.png'), bbox_inches='tight')
#"""
