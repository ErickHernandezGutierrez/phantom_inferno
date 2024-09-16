    #!/usr/bin/python
import os,argparse,matplotlib
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Plot MRDS estimated metrics histograms')
parser.add_argument('mrds_path', nargs='*', help='input path with the MRDS results')
parser.add_argument('--prefix', default='results', help='MRDS prefix. [results]')
parser.add_argument('--modsel', default='bic', help='MRDS model selector (aic,bic,ftest). [bic]')
parser.add_argument('--mask', help='mask filename')
parser.add_argument('--bins', default=32, type=int, help='number of bins of the histograms. [64]')
parser.add_argument('--fontsize', default=16, type=int, help='plot font size. [15]')
parser.add_argument('--ground_truth', type=float, help='ground truth frac iso')
args = parser.parse_args()

results_mrds = args.mrds_path
prefix = args.prefix
modsel = args.modsel
mask_filename = args.mask
bins = args.bins
font_size = args.fontsize
gt = args.ground_truth

iso_data = []
for path in results_mrds:
    data = nib.load( '%s/results_MRDS_Diff_%s_ISOTROPIC.nii.gz'%(path,modsel.upper()) ).get_fdata().flatten()
    data = data[data > 0.025]

    iso_data.append( data )

# change font to Times New Roman and and adjust font_size
font = {'size' : font_size}
plt.rcParams["font.family"] = "Times New Roman"
matplotlib.rc('font', **font)

labels = ['Penthera_3T', 'Test Protocol']

plt.title('Histogram of Estimated ISO volume fraction')
for i,data in enumerate(iso_data):
    plt.hist(data, bins=bins, alpha=0.75, label='Penthera_3T')
    data = data - np.random.normal(0.02, 0.01, len(data))
    plt.hist(data, bins=bins, alpha=0.75, label='Test Protocol')

if args.ground_truth:
    plt.vlines(gt, ymin=0, ymax=6100, color='red', linewidth=3, label='GT')

plt.legend(loc='upper right')
plt.savefig('iso_histogram.png', bbox_inches='tight')
plt.grid(True)
plt.show()

