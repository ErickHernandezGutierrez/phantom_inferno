#!/usr/bin/python
import os,argparse

parser = argparse.ArgumentParser(description='Track with iFOD2 using multi-tensor ODFs.')
parser.add_argument('study_path', help='path to the study folder')
parser.add_argument('scheme', help='protocol file in format Nx4 text file with [X Y Z b] at each row')
parser.add_argument('--modsel', default='bic', help='model selector (aic,bic,ftest). default: bic')
parser.add_argument('--mask', help='mask filename')
parser.add_argument('--nsubjects', default=1, type=int, help='number of subjects in the study. default: 1')
parser.add_argument('--ntracks', default=100000, type=int, help='number of selected streamlines for the tractogram. default: 100000')
args = parser.parse_args()

study = args.study_path
scheme = args.scheme
modsel = args.modsel
mask = args.mask
nsubjects = args.nsubjects
ntracks = args.ntracks

for i in range(nsubjects):
    subject = 'sub-%.3d_ses-1' % (i+1)
    print('├── Subject %s' % subject)

    print('│   ├── Fiber tracking')
    os.system('tckgen %s/%s/mrds/results_MRDS_Diff_%s_ODF_SH.nii.gz %s/%s/mrds/results_MRDS_Diff_%s_TRACKS.tck -select %d -seed_image %s -grad %s -force -quiet' % (study,subject,modsel.upper(),study,subject,modsel.upper(),ntracks,mask,scheme))