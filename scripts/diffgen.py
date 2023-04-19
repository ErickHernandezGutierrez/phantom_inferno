import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import sys, itertools, argparse, os

def lambdaFA2lambdaMD(lambda1, FA):
    a = FA / np.sqrt(3.0 - 2.0*FA*FA)

    MD = lambda1 / (1.0 + 2.0*a)
    lambda23 = MD * (1.0 - a)

    return lambda23

def lambdas2fa(lambda1, lambda23):
    a = np.sqrt(0.5)
    b = np.sqrt( (lambda1-lambda23)**2 + (lambda23-lambda23)**2 + (lambda23-lambda1)**2 )
    c = np.sqrt( lambda1**2 + lambda23**2 + lambda23**2 )
    
    return a*b/c

def lambdas2md(lambda1, lambda2):
    return (lambda1+lambda2+lambda2)/3.0

parser = argparse.ArgumentParser(description='Generate phantom lambdas (diffusivities).')
parser.add_argument('phantom', default='Training_3D_SF', help="phantom structure {Training_3D_SF,Training_SF,Training_X}. default: 'Training_3D_SF'")
parser.add_argument('model', default='multi-tensor', help="model for the phantom {multi-tensor,standard-model}. default: 'multi-tensor'")
parser.add_argument('study_path', help='path to the study folder')
parser.add_argument('--nsubjects', type=int, default=1, help='number of subjects to generate. default: 1')

parser.add_argument('--healthy_lambdas', type=float, default=[1.7e-3, 0.3e-3], nargs=2, help='diffusivities for healthy tissue. default: [1.7e-3, 0.3e-3] mm^2/s')
parser.add_argument('--damaged_lambdas', type=float, default=[1.7e-3, 0.3e-3], nargs=2, help='diffusivities for damaged tissue. default: [1.7e-3, 0.3e-3] mm^2/s')

#parser.add_argument('--d_par',  type=float, default=[1.7e-3, 1.7e-3], nargs=2, help='IC and EC parallel diffusivities. default: [1.7e-3, 1.7e-3] mm^2/s')
parser.add_argument('--d_par_ic',  type=float, default=1.7e-3, help='IC parallel diffusivity. default: 1.7e-3 mm^2/s')
parser.add_argument('--d_par_ec',  type=float, default=1.7e-3, help='EC parallel diffusivity. default: 1.7e-3 mm^2/s')
parser.add_argument('--d_perp_ec', type=float, default=0.5e-3, help='EC perpendicular diffusivity. default: 0.5e-3 mm^2/s')
parser.add_argument('--d_iso',     type=float, default=0.3e-3, help='ISO diffusivity. default: 0.3e-3 mm^2/s')
parser.add_argument('--size_ic',  type=float, default=0.7,  help='size of the intra-cellular compartment')
parser.add_argument('--size_ec',  type=float, default=0.25, help='size of the extra-cellular compartment')
parser.add_argument('--size_iso', type=float, default=0.05, help='size of the isotropic compartment')

parser.add_argument('--lesion_d_par', type=float, nargs=2, default=[1.7e-3, 1.7e-3], help='IC and EC parallel diffusivities (default: [1.7e-3, 1.7e-3] mm^2/s)')
parser.add_argument('--lesion_d_perp', type=float, default=[0.5e-3], help='EC perpendicular diffusivity (default: 0.5e-3 mm^2/s)')
parser.add_argument('--lesion_d_iso', type=float, default=[0.3e-3], help='ISO diffusivity (default: 0.3e-3 mm^2/s)')

parser.add_argument('--lesion_mask', nargs='*', default=['no-lesion-mask-1.nii', 'no-lesion-mask-2.nii', 'no-lesion-mask-3.nii'], help='lesion mask')
parser.add_argument('--healthy_mean', type=float, nargs=2, help='mean of diffusivities for healthy tissue')
parser.add_argument('--healthy_var',  type=float, nargs=2, help='variance of diffusivities for healthy tissue')
parser.add_argument('--lesion_mean',  type=float, nargs=2, help='mean of diffusivities for healthy tissue')
parser.add_argument('--lesion_var',   type=float, nargs=2, help='variance of diffusivities for healthy tissue')
parser.add_argument('--show_hist',  type=bool,  default=False, help='If True, show the histograms of the generated diffusivities')

args = parser.parse_args()

phantom = args.phantom
study = args.study_path
model = args.model

nsubjects = args.nsubjects

N = {
    'structures/Training_3D_SF': 3,
    'structures/Training_SF': 5,
    'structures/Training_X': 2
}

dims = {
    'structures/Training_3D_SF': [16,16,5],
    'structures/Training_SF': [16,16,5],
    'structures/Training_X': [32,32,10]
}

lesion_mask_filenames = args.lesion_mask

nbundles = N[phantom]

X,Y,Z = dims[phantom]

if args.healthy_mean and args.lesion_mean:
    healthy_lambda1_mean, healthy_lambda23_mean = args.healthy_mean
    damaged_lambda1_mean, damaged_lambda23_mean = args.lesion_mean

if args.healthy_var and args.lesion_var:
    healthy_lambda1_var, healthy_lambda23_var = args.healthy_var
    damaged_lambda1_var, damaged_lambda23_var = args.lesion_var

if args.healthy_lambdas and args.damaged_lambdas:
    lambda1  = np.array( [args.healthy_lambdas[0]]*(3*nsubjects) )
    lambda23 = np.array( [args.healthy_lambdas[1]]*(3*nsubjects) )

    lesion_lambda1  = np.array( [args.damaged_lambdas[0]]*(3*nsubjects) )
    lesion_lambda23 = np.array( [args.damaged_lambdas[1]]*(3*nsubjects) )

d_par_ic  = args.d_par_ic
d_par_ec  = args.d_par_ec
d_perp_ec = args.d_perp_ec
d_iso     = args.d_iso
size_ic   = args.size_ic
size_ec   = args.size_ec
size_iso  = args.size_iso

"""
healthy_lambda1_mean  = 1.38e-3
healthy_lambda23_mean = 2.50e-4

damaged_lambda1_mean  = 1.1e-3
damaged_lambda23_mean = 4.8e-4

healthy_lambda1_var = 3.37e-8
healthy_lambda23_var = 3.02e-9

damaged_lambda1_var = 2.37e-8
damaged_lambda23_var = 2.02e-9
"""

def plot_histograms(healthy_lambda1_mean, healthy_lambda23_mean, damaged_lambda1_mean, damaged_lambda23_mean):
    lambda1 = np.random.normal(loc=healthy_lambda1_mean,  scale=np.sqrt(healthy_lambda1_var),  size=10000)
    lambda23 = np.random.normal(loc=healthy_lambda23_mean, scale=np.sqrt(healthy_lambda23_var), size=10000)

    lesion_lambda1 = np.random.normal(loc=damaged_lambda1_mean,  scale=np.sqrt(damaged_lambda1_var),  size=10000)
    lesion_lambda23 = np.random.normal(loc=damaged_lambda23_mean, scale=np.sqrt(damaged_lambda23_var), size=10000)

    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20,10))

    data   = [ [lambda1,lambda23],[lesion_lambda1,lesion_lambda23] ]
    colors = [ 'blue','red' ]
    nbins = 16
    alpha = 0.75

    fig.suptitle('Distribution of Generated Diffusivities for Phantom')

    ax[0,0].set_ylabel('Healthy')
    ax[1,0].set_ylabel('Lesion')

    ax[0,0].set_title('Distribution of AD values')
    ax[0,1].set_title('Distribution of RD values')
    ax[0,2].set_title('Distribution of FA values')
    ax[0,3].set_title('Distribution of MD values')

    for i in range(2):
        lambdas = data[i]
        color = colors[i]

        fa = lambdas2fa(lambdas[0], lambdas[1])
        md = lambdas2md(lambdas[0], lambdas[1])

        ax[i,0].hist(lambdas[0]*1e3, bins=nbins, color=color, edgecolor='black', alpha=alpha)
        ax[i,0].set_xlabel('AD [$\mu m^2/ms$]') #(r'$\lambda_{1}$ [$\mu m^2/ms$]')
        ax[i,0].set_xlim([0.5, 3.1])

        ax[i,1].hist(lambdas[1]*1e3, bins=nbins, color=color, edgecolor='black', alpha=alpha)
        ax[i,1].set_xlabel('RD [$\mu m^2/ms$]') #(r'$\lambda_{2,3}$ [$\mu m^2/ms$]')
        ax[i,1].set_xlim([0.0, 1.0])

        ax[i,2].hist(fa, bins=nbins, color=color, edgecolor='black', alpha=alpha)
        ax[i,2].set_xlabel('FA')
        ax[i,2].set_xlim([0, 1])

        ax[i,3].hist(md*1e3, bins=nbins, color=color, edgecolor='black', alpha=alpha)
        ax[i,3].set_xlabel(r'MD [$\mu m^2/ms$]')
        ax[i,3].set_xlim([0.4, 1.4])
        
    #fig.savefig('histogram-lambdas.png', bbox_inches='tight')
    plt.show()

# Load WM and lesion masks
mask = np.zeros((X,Y,Z, nbundles), dtype=np.uint8)
lesion_mask = np.zeros((X,Y,Z, nbundles), dtype=np.uint8)
for i in range(nbundles):
    mask[:,:,:, i] = nib.load('%s/wm-mask-%d.nii' % (phantom,i+1)).get_fdata()
    lesion_mask[:,:,:, i] = nib.load('%s/%s' % (phantom,lesion_mask_filenames[i])).get_fdata()

if args.healthy_mean and args.lesion_mean:
    # Plot diffusivities distributions
    if args.show_hist:
        plot_histograms(healthy_lambda1_mean, healthy_lambda23_mean, damaged_lambda1_mean, damaged_lambda23_mean)

    # GT lambda values
    lambda1 = np.random.normal(loc=healthy_lambda1_mean,  scale=np.sqrt(healthy_lambda1_var),  size=3*nsubjects)
    lambda23 = np.random.normal(loc=healthy_lambda23_mean, scale=np.sqrt(healthy_lambda23_var), size=3*nsubjects)

    lesion_lambda1 = np.random.normal(loc=damaged_lambda1_mean,  scale=np.sqrt(damaged_lambda1_var),  size=3*nsubjects)
    lesion_lambda23 = np.random.normal(loc=damaged_lambda23_mean, scale=np.sqrt(damaged_lambda23_var), size=3*nsubjects)

# ground-truth FA and MD
healthy_FA = lambdas2fa(lambda1, lambda23)
healthy_MD = lambdas2md(lambda1, lambda23)

damaged_FA = lambdas2fa(lesion_lambda1, lesion_lambda23)
damaged_MD = lambdas2md(lesion_lambda1, lesion_lambda23)

# repeat the lambdas in all the voxels
lambda1 = np.repeat(lambda1, X*Y*Z)
lambda23 = np.repeat(lambda23, X*Y*Z)
healthy_FA = np.repeat(healthy_FA, X*Y*Z)
healthy_MD = np.repeat(healthy_MD, X*Y*Z)

lesion_lambda1 = np.repeat(lesion_lambda1, X*Y*Z)
lesion_lambda23 = np.repeat(lesion_lambda23, X*Y*Z)
damaged_FA = np.repeat(damaged_FA, X*Y*Z)
damaged_MD = np.repeat(damaged_MD, X*Y*Z)

# reshape
lambda1 = lambda1.reshape(nsubjects,3,X,Y,Z)
lambda23 = lambda23.reshape(nsubjects,3,X,Y,Z)
healthy_FA = healthy_FA.reshape(nsubjects,3,X,Y,Z)
healthy_MD = healthy_MD.reshape(nsubjects,3,X,Y,Z)

lesion_lambda1 = lesion_lambda1.reshape(nsubjects,3,X,Y,Z)
lesion_lambda23 = lesion_lambda23.reshape(nsubjects,3,X,Y,Z)
damaged_FA = damaged_FA.reshape(nsubjects,3,X,Y,Z)
damaged_MD = damaged_MD.reshape(nsubjects,3,X,Y,Z)

if not os.path.exists( study ):
    os.system( 'mkdir %s' % study )

for sub_id in range(nsubjects):
    subject = '%s/sub-%.3d_ses-1' % (study,sub_id+1)

    print('generating lambdas for %s from phantom %s' % (subject,phantom))

    lambdas = np.zeros((X,Y,Z, 3*nbundles))
    ad = np.zeros((X,Y,Z, nbundles))
    rd = np.zeros((X,Y,Z, nbundles))
    fa = np.zeros((X,Y,Z, nbundles))
    md = np.zeros((X,Y,Z, nbundles))

    for bundle_id in range(nbundles):
        lambdas[:,:,:, 3*bundle_id]   += lambda1[sub_id, bundle_id, :,:,:] * (mask[:,:,:, bundle_id]-lesion_mask[:,:,:, bundle_id])
        lambdas[:,:,:, 3*bundle_id+1] += lambda23[sub_id, bundle_id, :,:,:] * (mask[:,:,:, bundle_id]-lesion_mask[:,:,:, bundle_id])
        lambdas[:,:,:, 3*bundle_id+2] += lambda23[sub_id, bundle_id, :,:,:] * (mask[:,:,:, bundle_id]-lesion_mask[:,:,:, bundle_id])

        lambdas[:,:,:, 3*bundle_id]   += lesion_lambda1[sub_id, bundle_id, :,:,:] * (lesion_mask[:,:,:, bundle_id])
        lambdas[:,:,:, 3*bundle_id+1] += lesion_lambda23[sub_id, bundle_id, :,:,:] * (lesion_mask[:,:,:, bundle_id])
        lambdas[:,:,:, 3*bundle_id+2] += lesion_lambda23[sub_id, bundle_id, :,:,:] * (lesion_mask[:,:,:, bundle_id])

        # ground-truth metrics
        ad[:,:,:, bundle_id] = lambdas[:,:,:, 3*bundle_id]
        rd[:,:,:, bundle_id] = lambdas[:,:,:, 3*bundle_id+1]

        fa[:,:,:, bundle_id] += healthy_FA[sub_id, bundle_id, :,:,:] * (mask[:,:,:, bundle_id]-lesion_mask[:,:,:, bundle_id])
        fa[:,:,:, bundle_id] += damaged_FA[sub_id, bundle_id, :,:,:] * (lesion_mask[:,:,:, bundle_id])

        md[:,:,:, bundle_id] += healthy_MD[sub_id, bundle_id, :,:,:] * (mask[:,:,:, bundle_id]-lesion_mask[:,:,:, bundle_id])
        md[:,:,:, bundle_id] += damaged_MD[sub_id, bundle_id, :,:,:] * (lesion_mask[:,:,:, bundle_id])

    # create subject folders
    if not os.path.exists( subject ):
        os.system( 'mkdir %s' % (subject) )
    if not os.path.exists( '%s/ground_truth' % (subject) ):
        os.system( 'mkdir %s/ground_truth' % (subject) )

    # save lambdas and metrics
    nib.save( nib.Nifti1Image(lambdas, np.identity(4)), '%s/ground_truth/lambdas.nii'%(subject) ) 
    nib.save( nib.Nifti1Image(ad, np.identity(4)), '%s/ground_truth/ad.nii'%(subject) ) 
    nib.save( nib.Nifti1Image(rd, np.identity(4)), '%s/ground_truth/rd.nii'%(subject) ) 
    nib.save( nib.Nifti1Image(fa, np.identity(4)), '%s/ground_truth/fa.nii'%(subject) ) 
    nib.save( nib.Nifti1Image(md, np.identity(4)), '%s/ground_truth/md.nii'%(subject) ) 
