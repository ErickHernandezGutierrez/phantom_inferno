#!/usr/bin/python
import os,argparse,matplotlib
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Plot MRDS estimated metrics histograms')
parser.add_argument('mrds_path', help='input path with the MRDS results')
parser.add_argument('--prefix', default='results', help='MRDS prefix. [results]')
parser.add_argument('--modsel', default='bic', help='MRDS model selector (aic,bic,ftest). [bic]')
parser.add_argument('--mask', help='mask filename')
parser.add_argument('--bins', default=32, type=int, help='number of bins of the histograms. [64]')
parser.add_argument('--ground_truth', type=float, help='ground truth frac iso')
args = parser.parse_args()

results_mrds = args.mrds_path
prefix = args.prefix
modsel = args.modsel
mask_filename = args.mask
bins = args.bins
gt = args.ground_truth

iso = nib.load( '%s/results_MRDS_Diff_%s_ISOTROPIC.nii.gz'%(results_mrds,modsel.upper()) ).get_fdata().flatten()
iso = iso[iso > 0.025]

font_size = 15
font = {'size' : font_size}
matplotlib.rc('font', **font)

plt.title('Histogram of Estimated ISO volume fraction')
plt.hist(iso, bins=bins, label='MRDS')
if args.ground_truth:
    plt.vlines(gt, ymin=0, ymax=10000, color='red', linewidth=3, label='GT')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
