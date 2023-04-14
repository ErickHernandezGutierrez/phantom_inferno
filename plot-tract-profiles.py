import matplotlib.pyplot as plt
import numpy as np
import json, sys, matplotlib
from utils import lambdas2fa, lambdas2md


json_filename = sys.argv[1]

json_file = open( json_filename )

data = json.load( json_file )

font = {'size' : 16}
bins = 16

#ad_gt = np.array([1.38, 1.38, 1.38, 1.38, 1.38, 1.38, 1.38, 1.38, 1.38, 1.38, 1.38, 1.38, 1.38, 1.38, 1.38, 1.38, 1.38, 1.38, 1.38, 1.38])
ad_gt = np.array([1.38, 1.38, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.38, 1.38])
#rd_gt = np.array([2.80, 2.80, 2.80, 2.80, 2.80, 2.80, 2.80, 2.80, 2.80, 2.80, 2.80, 2.80, 2.80, 2.80, 2.80, 2.80, 2.80, 2.80, 2.80, 2.80]) * 1e-1
rd_gt = np.array([2.80, 2.80, 6.00, 6.00, 6.00, 6.00, 6.00, 6.00, 6.00, 6.00, 6.00, 6.00, 6.00, 6.00, 6.00, 6.00, 6.00, 6.00, 2.80, 2.80]) * 1e-1
fa_gt = lambdas2fa( [ad_gt, rd_gt, rd_gt] )
md_gt = lambdas2md( [ad_gt, rd_gt, rd_gt] )
gt = [ad_gt, rd_gt, fa_gt, md_gt]

matplotlib.rc('font', **font)

profiles = [
#   AD  RD  FA  MD
    [], [], [], [], # bundle 1
    [], [], [], [], # bundle 2
    [], [], [], []  # bundle 3
]

colors = {
    'bundle-1_ic_ad_metric' : '#7E7EFD',
    'bundle-1_ic_rd_metric' : '#7E7EFD',
    'bundle-1_ic_fa_metric' : '#7E7EFD',
    'bundle-1_ic_md_metric' : '#7E7EFD',
    'bundle-2_ic_ad_metric' : '#7E7EFD',
    'bundle-2_ic_rd_metric' : '#7E7EFD',
    'bundle-2_ic_fa_metric' : '#7E7EFD',
    'bundle-2_ic_md_metric' : '#7E7EFD',
    'bundle-3_ic_ad_metric' : '#7E7EFD',
    'bundle-3_ic_rd_metric' : '#7E7EFD',
    'bundle-3_ic_fa_metric' : '#7E7EFD',
    'bundle-3_ic_md_metric' : '#7E7EFD',
    'ad_metric' : '#7FBF7F',
    'rd_metric' : '#7FBF7F',
    'fa_metric' : '#7FBF7F',
    'md_metric' : '#7FBF7F'
}

labels = {
    'bundle-1_ic_ad_metric' : 'fixel-AD',
    'bundle-1_ic_rd_metric' : 'fixel-RD',
    'bundle-1_ic_fa_metric' : 'fixel-FA',
    'bundle-1_ic_md_metric' : 'fixel-MD',
    'bundle-2_ic_ad_metric' : 'fixel-AD',
    'bundle-2_ic_rd_metric' : 'fixel-RD',
    'bundle-2_ic_fa_metric' : 'fixel-FA',
    'bundle-2_ic_md_metric' : 'fixel-MD',
    'bundle-3_ic_ad_metric' : 'fixel-AD',
    'bundle-3_ic_rd_metric' : 'fixel-RD',
    'bundle-3_ic_fa_metric' : 'fixel-FA',
    'bundle-3_ic_md_metric' : 'fixel-MD',
    'ad_metric' : 'AD',
    'rd_metric' : 'RD',
    'fa_metric' : 'FA',
    'md_metric' : 'MD',
    '0' : 'GT AD',
    '1' : 'GT RD',
    '2' : 'GT FA',
    '3' : 'GT MD'
}

ylabels = [r'AD [$\mu m^2/ms$]', r'RD [$\mu m^2/ms$]', 'FA', r'MD [$\mu m^2/ms$]']

coords = {
    # bundles
    'bundle-1': 0,
    'bundle-2': 1,
    'bundle-3': 2,

    # metrics
    'ad_metric' : 0,
    'rd_metric' : 1,
    'fa_metric' : 2,
    'md_metric' : 3,
    'bundle-1_ic_ad_metric' : 0,
    'bundle-1_ic_rd_metric' : 1,
    'bundle-1_ic_fa_metric' : 2,
    'bundle-1_ic_md_metric' : 3,
    'bundle-2_ic_ad_metric' : 0,
    'bundle-2_ic_rd_metric' : 1,
    'bundle-2_ic_fa_metric' : 2,
    'bundle-2_ic_md_metric' : 3,
    'bundle-3_ic_ad_metric' : 0,
    'bundle-3_ic_rd_metric' : 1,
    'bundle-3_ic_fa_metric' : 2,
    'bundle-3_ic_md_metric' : 3
}

ylims = [
    [0.00110e3, 0.00155e3],
    [0.00022e3, 0.00040e3],
    [0.55, 0.8],
    [0.00058e3, 0.00072e3]
]

xmax = [21, 21, 14]
xticks = [ [1, 5, 10, 15, 20], [1, 5, 10, 15, 20], np.arange(1,14,2) ]

fig, ax = plt.subplots(nrows=3, ncols=4)

for bundle in data:
    if bundle != 'bundle-3':

        i = coords[bundle]

        dimx = np.arange(1,xmax[i])

        for j in range(4):
            #ax[i,j].hlines(y=gt[j], xmin=1, xmax=xmax[i]-1, linewidth=3, color='red', label=labels[str(j)])
            ax[i,j].plot(dimx, gt[j], linewidth=3, color='red', label=labels[str(j)])

        for metric in data[bundle]:
            if (bundle in metric) or (not 'bundle' in metric):
                
                j = coords[metric]

                profile_mean = []
                profile_var  = []

                for point in data[bundle][metric]:
                    # mean and std of the subjects
                    m = np.average( data[bundle][metric][point]['mean'] )
                    s = np.average( data[bundle][metric][point]['std']  )

                    profile_mean.append( m )
                    profile_var.append( s )

                profile_mean = np.array(profile_mean)
                profile_var  = np.array(profile_var)

                if not 'fa' in metric:
                    profile_mean *= 1e3
                    profile_var  *= 1e3

                ax[i,j].fill_between(dimx, profile_mean-profile_var, profile_mean+profile_var, color=colors[metric], label=labels[metric])
                ax[i,j].plot(dimx, profile_mean, color='black')
                ax[i,j].set_xlabel('Location along the streamline')
                ax[i,j].set_ylabel( ylabels[j] )
                ax[i,j].set_xticks( xticks[i] )
                ax[i,j].set_ylim( ylims[j] )
                ax[i,j].legend(loc='upper left', prop={'size': 10})

plt.show()
