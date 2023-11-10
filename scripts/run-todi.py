import os, sys, argparse

parser = argparse.ArgumentParser(description='Run TODI.')
parser.add_argument('tractogram_filename', help='input tractogram filename')
parser.add_argument('dwi_filename', help='input DWI filename for anatomical reference')
parser.add_argument('todf_filename', help='output tODF filename')
parser.add_argument('fixel_path', help='ouput path for the fixel folder')

parser.add_argument('--lmax', type=int, default=16, help='lmax for the tODFs. Default: 16')
parser.add_argument('--peak_threshold', type=float, default=0.1, help='threshold peak amplitude of positive FOD lobes. Default: 0.1')

args = parser.parse_args()

tractogram = args.tractogram_filename
lmax = args.lmax
dwi = args.dwi_filename
todf = args.todf_filename
fixel = args.fixel_path
threshold = args.peak_threshold

os.system('tckmap %s %s -tod %d -template %s -force' % (tractogram,todf,lmax,dwi))

os.system('rm -r %s' % fixel)
os.system('mkdir %s' % fixel)

os.system('fod2fixel %s %s -nii -maxnum 3 -fmls_peak_value %f -afd afd.nii -peak_amp peaks.nii -force' % (todf,fixel,threshold))

os.system('fixel2voxel %s/directions.nii count %s/nufo.nii' % (fixel,fixel))
os.system('fixel2voxel %s/afd.nii none %s/afd_none.nii' % (fixel,fixel))
os.system('fixel2voxel %s/peaks.nii none %s/peaks_none.nii' % (fixel,fixel))
