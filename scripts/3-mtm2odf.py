#!/usr/bin/python
import os,argparse

parser = argparse.ArgumentParser(description='Convert Multi-Tensor Field (MTF) to Orientation Distribution Function (ODF) image.')
parser.add_argument('study_path', help='path to the study folder')
parser.add_argument('scheme', help='protocol file in format Nx4 text file with [X Y Z b] at each row')
parser.add_argument('--modsel', default='bic', help='model selector (aic,bic,ftest). default: bic')
parser.add_argument('--mask', help='mask filename')
parser.add_argument('--nsubjects', default=1, type=int, help='number of subjects in the study. default: 1')
args = parser.parse_args()

study = args.study_path
scheme = args.scheme
modsel = args.modsel
mask = args.mask
nsubjects = args.nsubjects

print('\n│Pipeline - Step 3 (Multi-Tensors to ODFs)│')

for i in range(nsubjects):
    subject = 'sub-%.3d_ses-1' % (i+1)
    print('├── Subject %s' % subject)

    print('│   ├── Converting MTF to ODFs')
    os.system('mdtodf %s/%s/mrds/results_MRDS_Diff_%s_COMP_SIZE.nii.gz %s/%s/mrds/results_MRDS_Diff_%s_PDDs_CARTESIAN.nii.gz %s/%s/mrds/results_MRDS_Diff_%s_ODF -eig %s/%s/mrds/results_MRDS_Diff_%s_EIGENVALUES.nii.gz -mask %s -minmax > %s/%s/mrds/mdtodf_log.txt' % (study,subject,modsel.upper(),study,subject,modsel.upper(),study,subject,modsel.upper(),study,subject,modsel.upper(),mask,study,subject))
