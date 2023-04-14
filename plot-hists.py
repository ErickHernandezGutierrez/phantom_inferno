import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.stats as st
import matplotlib, sys, os

experiment = sys.argv[1]
nsubjects = int(sys.argv[2])
output_path = sys.argv[3]

def plot_hist(arr, color, ax, label):
    n, bins, patches = ax.hist(arr, bins=64, density=1, facecolor=color, alpha=0.0)

    (mu, sigma) = st.norm.fit(arr)
    y = st.norm.pdf(bins, mu, sigma)

    ax.plot(bins, y, color=color, linewidth=2, label=label)

def lambdas2fa(L1, L2):
    a = np.sqrt(0.5)
    b = np.sqrt( (L1-L2)**2 + (L2-L2)**2 + (L2-L1)**2 )
    c = np.sqrt( L1**2 + L2**2 + L2**2 )

    return a*b/c

def lambdas2md(L1, L2):
    return (L1 + L2 + L2) / 3

def relative_error(actual, gt):
    expected = np.repeat( gt, actual.shape[0] )

    err = np.abs( (actual-expected)/expected )

    return np.mean(err), np.var(err)

gt_ad = [[],[],[]]
gt_rd = [[],[],[]]
gt_fa = [[],[],[]]
gt_md = [[],[],[]]

fixel_ad = [[],[],[]]
fixel_rd = [[],[],[]]
fixel_fa = [[],[],[]]
fixel_md = [[],[],[]]

ad = [[],[],[]]
rd = [[],[],[]]
fa = [[],[],[]]
md = [[],[],[]]

for sub_id in range(1,nsubjects+1):
    subject = 'sub-%.3d_ses-1' % sub_id

    gt_ad = nib.load('%s/%s/ground_truth/ad.nii' % (experiment,subject)).get_fdata()
    gt_rd = nib.load('%s/%s/ground_truth/rd.nii' % (experiment,subject)).get_fdata()
    gt_fa = nib.load('%s/%s/ground_truth/fa.nii' % (experiment,subject)).get_fdata()
    gt_md = nib.load('%s/%s/ground_truth/md.nii' % (experiment,subject)).get_fdata()

    for i in range(3):
        fixel_ad[i] += nib.load('%s/results_tractometry/%s/Fixel_MRDS_Along_Bundle/bundle-%d_ic_ad_metric.nii.gz'%(experiment,subject,(i+1))).get_fdata().flatten().tolist()
        fixel_rd[i] += nib.load('%s/results_tractometry/%s/Fixel_MRDS_Along_Bundle/bundle-%d_ic_rd_metric.nii.gz'%(experiment,subject,(i+1))).get_fdata().flatten().tolist()
        fixel_fa[i] += nib.load('%s/results_tractometry/%s/Fixel_MRDS_Along_Bundle/bundle-%d_ic_fa_metric.nii.gz'%(experiment,subject,(i+1))).get_fdata().flatten().tolist()
        fixel_md[i] += nib.load('%s/results_tractometry/%s/Fixel_MRDS_Along_Bundle/bundle-%d_ic_md_metric.nii.gz'%(experiment,subject,(i+1))).get_fdata().flatten().tolist()

        ad[i] += nib.load('%s/results_tractometry/%s/bundle-%d_ad_metric.nii'%(experiment,subject,(i+1))).get_fdata().flatten().tolist()
        rd[i] += nib.load('%s/results_tractometry/%s/bundle-%d_rd_metric.nii'%(experiment,subject,(i+1))).get_fdata().flatten().tolist()
        fa[i] += nib.load('%s/results_tractometry/%s/bundle-%d_fa_metric.nii'%(experiment,subject,(i+1))).get_fdata().flatten().tolist()
        md[i] += nib.load('%s/results_tractometry/%s/bundle-%d_md_metric.nii'%(experiment,subject,(i+1))).get_fdata().flatten().tolist()

for i in range(3):
    fixel_ad[i] = np.array( list(filter((0.0).__ne__, fixel_ad[i])) ) * 1e3
    fixel_rd[i] = np.array( list(filter((0.0).__ne__, fixel_rd[i])) ) * 1e3
    fixel_fa[i] = np.array( list(filter((0.0).__ne__, fixel_fa[i])) )
    fixel_md[i] = np.array( list(filter((0.0).__ne__, fixel_md[i])) ) * 1e3

    ad[i] = np.array( list(filter((0.0).__ne__, ad[i])) ) * 1e3
    rd[i] = np.array( list(filter((0.0).__ne__, rd[i])) ) * 1e3
    fa[i] = np.array( list(filter((0.0).__ne__, fa[i])) )
    md[i] = np.array( list(filter((0.0).__ne__, md[i])) ) * 1e3

"""healthy_lambda1  = 2.50
healthy_lambda23 = 2.80e-1
damaged_lambda1  = 1.75
damaged_lambda23 = 7.00e-1
var_lambda1 = 2.37e-2
var_lambda23 = 3.02e-3"""

healthy_lambda1  = 1.38e-3 * (1e3)
healthy_lambda23 = 2.50e-4 * (1e3)

damaged_lambda1  = 1.1e-3 * (1e3)
damaged_lambda23 = 4.8e-4 * (1e3)

var_lambda1  = 3.37e-8 * (1e6)
var_lambda23 = 3.02e-9 * (1e6)

var_damaged_lambda1  = 2.37e-8 * (1e6)
var_damaged_lambda23 = 2.02e-9 * (1e6)

L1 = np.random.normal(loc=healthy_lambda1,  scale=np.sqrt(var_lambda1),  size=10000)
L2 = np.random.normal(loc=healthy_lambda23, scale=np.sqrt(var_lambda23), size=10000)

D1 = np.random.normal(loc=damaged_lambda1,  scale=np.sqrt(var_damaged_lambda1),  size=10000)
D2 = np.random.normal(loc=damaged_lambda23, scale=np.sqrt(var_damaged_lambda23), size=10000)

FA = lambdas2fa(L1, L2)
MD = lambdas2md(L1, L2)

FAD = lambdas2fa(D1, D2)
MDD = lambdas2md(D1, D2)

font = {'size' : 20}
matplotlib.rc('font', **font)
bins = 16

xlabels = {
    'AD': r'AD [$\mu m^2/ms$]',
    'RD': r'RD [$\mu m^2/ms$]',
    'FA': 'FA',
    'MD': r'MD [$\mu m^2/ms$]'
}

xlims = {
    'AD': [0.51, 2.01],
    'RD': [0.01, 0.61],
    'FA': [0.31, 1.01],
    'MD': [0.51, 0.81]
}

ymax = {
    ('bundle-1','AD'): 7,
    ('bundle-1','RD'): 19,
    ('bundle-1','FA'): 19,
    ('bundle-1','MD'): 25,

    ('bundle-2','AD'): 9,
    ('bundle-2','RD'): 19,
    ('bundle-2','FA'): 19,
    ('bundle-2','MD'): 45,

    ('bundle-3','AD'): 7,
    ('bundle-3','RD'): 19,
    ('bundle-3','FA'): 15,
    ('bundle-3','MD'): 25
}

"""gt = {
    'AD': L1,
    'RD': L2,
    'FA': FA,
    'MD': MD
}"""

gt = {
    ('bundle-1','AD'): gt_ad[4,8,2, 0]*1e3,
    ('bundle-1','RD'): gt_rd[4,8,2, 0]*1e3,
    ('bundle-1','FA'): gt_fa[4,8,2, 0]    ,
    ('bundle-1','MD'): gt_md[4,8,2, 0]*1e3,

    ('bundle-2','AD'): gt_ad[0,15,2, 1]*1e3,
    ('bundle-2','RD'): gt_rd[0,15,2, 1]*1e3,
    ('bundle-2','FA'): gt_fa[0,15,2, 1]    ,
    ('bundle-2','MD'): gt_md[0,15,2, 1]*1e3,

    ('bundle-3','AD'): gt_ad[4,8,2, 2]*1e3,
    ('bundle-3','RD'): gt_rd[4,8,2, 2]*1e3,
    ('bundle-3','FA'): gt_fa[4,8,2, 2]    ,
    ('bundle-3','MD'): gt_md[4,8,2, 2]*1e3
}

histograms = {
    ('bundle-1', 'AD'): ad[0],
    ('bundle-1', 'RD'): rd[0],
    ('bundle-1', 'FA'): fa[0],
    ('bundle-1', 'MD'): md[0],

    ('bundle-1', 'fixel-AD'): fixel_ad[0],
    ('bundle-1', 'fixel-RD'): fixel_rd[0],
    ('bundle-1', 'fixel-FA'): fixel_fa[0],
    ('bundle-1', 'fixel-MD'): fixel_md[0],

    ('bundle-2', 'AD'): ad[1],
    ('bundle-2', 'RD'): rd[1],
    ('bundle-2', 'FA'): fa[1],
    ('bundle-2', 'MD'): md[1],

    ('bundle-2', 'fixel-AD'): fixel_ad[1],
    ('bundle-2', 'fixel-RD'): fixel_rd[1],
    ('bundle-2', 'fixel-FA'): fixel_fa[1],
    ('bundle-2', 'fixel-MD'): fixel_md[1],

    ('bundle-3', 'AD'): ad[2],
    ('bundle-3', 'RD'): rd[2],
    ('bundle-3', 'FA'): fa[2],
    ('bundle-3', 'MD'): md[2],

    ('bundle-3', 'fixel-AD'): fixel_ad[2],
    ('bundle-3', 'fixel-RD'): fixel_rd[2],
    ('bundle-3', 'fixel-FA'): fixel_fa[2],
    ('bundle-3', 'fixel-MD'): fixel_md[2],
}

color = '#7FBF7F'
fixel_color = '#7E7EFD'

fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(16,9))
for i,bundle in enumerate(['bundle-1', 'bundle-2', 'bundle-3']):
    for j,metric in enumerate(['AD', 'RD', 'FA', 'MD']):
        print( (bundle, metric) )

        ax[i,j].hist(histograms[(bundle,metric)], bins=bins, edgecolor=color, color=color, density=True, alpha=0.55, label=metric)
        ax[i,j].hist(histograms[(bundle,'fixel-'+metric)], bins=bins, edgecolor=fixel_color, color=fixel_color, density=True, alpha=0.55, label='fixel-'+metric)

        dti_err = relative_error(histograms[(bundle,metric)], gt[bundle,metric])        
        mtm_err = relative_error(histograms[(bundle,'fixel-'+metric)], gt[bundle,metric])

        print('DTI: %f ± %f' % (dti_err[0], dti_err[1]))
        print('MTM: %f ± %f' % (mtm_err[0], mtm_err[1]))

        ax[i,j].set_xlabel( xlabels[metric] )
        ax[i,j].set_xlim( xlims[metric] )
        ax[i,j].set_facecolor('#E5E5E5')

        ax[i,j].vlines(gt[bundle,metric], ymin=0, ymax=ymax[bundle,metric], linewidth=4, color='red')

        #plot_hist(gt[metric], 'red', ax[i,j], 'GT '+metric)

#plt.close(fig)
#fig.savefig(os.path.join(output_path, 'histograms.png'), bbox_inches='tight')
plt.show()

"""
for i in range(3):
    ax[i,0].hist(fixel_ad[i], bins=bins, edgecolor='black', color='blue',  density=True, alpha=0.5, label='fixel-AD')
    #ax[i,0].hist(ad[i],       bins=bins, edgecolor='black', color='green', density=True, alpha=0.5, label='AD')
    #if i==0:
        #ax[i,0].set_title('Histogram of AD values')
        #ax[i,0].set_ylabel('Bundle 1')
    #if i==1:
        #ax[i,0].set_ylabel('Bundle 2')
    #if i==2:
        #ax[i,0].set_ylabel('Bundle 3')
    ax[i,0].set_xlabel(r'AD [$\mu m^2/ms$]')
    if i == 1:
        plot_hist(L1, 'red', ax[i,0], 'GT AD')
    else:
        plot_hist(L1, 'red', ax[i,0], 'GT AD')
    ax[i,0].legend(loc='upper left', prop={'size': 10})

    ax[i,1].hist(fixel_rd[i], bins=bins, edgecolor='black', color='blue',  density=True, alpha=0.5, label='fixel-RD')
    #ax[i,1].hist(rd[i],       bins=bins, edgecolor='black', color='green', density=True, alpha=0.5, label='RD')
    #if i==0:
        #ax[i,1].set_title('Histogram of RD values')
    #if i==2:
    ax[i,1].set_xlabel(r'RD [$\mu m^2/ms$]')
    if i == 1:
        plot_hist(L2, 'red', ax[i,1], 'GT RD')
    else:
        plot_hist(L2, 'red', ax[i,1], 'GT RD')
    ax[i,1].legend(loc='upper left', prop={'size': 10})

    ax[i,2].hist(fixel_fa[i], bins=bins, edgecolor='black', color='blue',  density=True, alpha=0.5, label='fixel-FA')
    #ax[i,2].hist(fa[i],       bins=bins, edgecolor='black', color='green', density=True, alpha=0.5, label='FA')
    #if i==0:
        #ax[i,2].set_title('Histogram of FA values')
    #if i==2:
    ax[i,2].set_xlabel('FA')
    ax[i,2].set_xlim([0, 1])
    if i == 1:
        plot_hist(FA, 'red', ax[i,2], 'GT FA')
    else:
        plot_hist(FA, 'red', ax[i,2], 'GT FA')
    ax[i,2].legend(loc='upper left', prop={'size': 10})

    ax[i,3].hist(fixel_md[i], bins=bins, edgecolor='black', color='blue',  density=True, alpha=0.5, label='fixel-MD')
    #ax[i,3].hist(md[i],       bins=bins, edgecolor='black', color='green', density=True, alpha=0.5, label='MD')
    #if i==0:
        #ax[i,3].set_title('Histogram of MD values')
    #if i==2:
    ax[i,3].set_xlabel(r'MD [$\mu m^2/ms$]')
    if i == 1:
        plot_hist(MD, 'red', ax[i,3], 'GT MD')
    else:
        plot_hist(MD, 'red', ax[i,3], 'GT MD')
    ax[i,3].legend(loc='upper left', prop={'size': 10})
"""
