import os, argparse

parser = argparse.ArgumentParser(description='Run TODI and use it as model selector for MRDS.')
parser.add_argument('in_mrds', help='Path to the MRDS results.')
parser.add_argument('in_tractogram', help='Tractogram filename. Format must be tck.')
parser.add_argument('in_dwi', help='DWI file for anatomical reference.')

parser.add_argument('out_todf', help='Output tODF.')
parser.add_argument('out_path', help='Ouput path for the fixel folder')

parser.add_argument('--prefix', default='results', help='Prefix of the MRDS results. [results]')
parser.add_argument('--method', default='Diff', help='Method used when estimating the fixel diffusivities in MRDS (Fixed,Equal,Diff). [Diff]')
parser.add_argument('--modsel', default='bic', help='Model selector (aic,bic,ftest). [bic]')

parser.add_argument('--lmax', type=int, default=16, help='lmax for the tODFs. [16]')
parser.add_argument('--peak_threshold', type=float, default=0.1, help='Threshold peak amplitude of positive FOD lobes. [0.1]')

parser.add_argument('-directions', help='Compute directions from the NuFO', action='store_true')
parser.add_argument('-afd_none', help='Compute AFD none', action='store_true')
parser.add_argument('-peaks_none', help='Compute peaks none', action='store_true')

args = parser.parse_args()

in_mrds = args.in_mrds
in_tractogram = args.in_tractogram
in_dwi = args.in_dwi

out_todf = args.out_todf
out_path = args.out_path

lmax = args.lmax
threshold = args.peak_threshold

os.system('tckmap %s %s -tod %d -template %s -force' % (in_tractogram,out_todf,lmax,in_dwi))

if not os.path.exists( out_path ):
    os.system('rm -rf %s' % out_path)
    os.system('mkdir %s' % out_path)

os.system('fod2fixel %s %s -nii -maxnum 3 -fmls_peak_value %f -afd afd.nii -peak_amp peaks.nii -force' % (out_todf,out_path,threshold))

if args.directions:
    os.system('fixel2voxel %s/directions.nii count %s/nufo.nii' % (out_path,out_path))
if args.afd_none:
    os.system('fixel2voxel %s/afd.nii none %s/afd_none.nii' % (out_path,out_path))
if args.peaks_none:
    os.system('fixel2voxel %s/peaks.nii none %s/peaks_none.nii' % (out_path,out_path))

