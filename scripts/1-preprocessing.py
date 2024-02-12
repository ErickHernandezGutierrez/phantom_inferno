#!/usr/bin/python
import os,argparse

parser = argparse.ArgumentParser(description='Denoise phantom signal.')
parser.add_argument('study_path', help='path to the study folder')
parser.add_argument('--nsubjects', default=1, type=int, help='number of subjects in the study. default: 1')
args = parser.parse_args()

study = args.study_path
nsubjects = args.nsubjects

print('\n│Pipeline - Step 1 (Preprocessing Data)│')

for i in range(nsubjects):
    subject = 'sub-%.3d_ses-1' % (i+1)
    print('├── Subject %s' % subject)

    print('│   ├── Denoising')
    os.system('dwidenoise %s/%s/dwi.nii.gz %s/%s/denoised_dwi.nii.gz -noise %s/%s/sigmas.nii.gz -force -quiet' % (study,subject,study,subject,study,subject))

    print('│   ├── Rician Bias Correction (Gudbjartsson)')
    os.system('mrcalc %s/%s/denoised_dwi.nii.gz 2 -pow %s/%s/sigmas.nii.gz 2 -pow 2 -multiply -sub 0.001 -max -sqrt %s/%s/corrected_dwi.nii.gz -force -quiet' % (study,subject,study,subject,study,subject))
