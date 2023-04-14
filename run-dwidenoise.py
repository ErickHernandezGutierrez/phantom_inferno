#!/usr/bin/python
import os,sys,argparse

parser = argparse.ArgumentParser(description='Denoise phantom signal.')
parser.add_argument('--study_path', help='path to the study folder')
parser.add_argument('--nsubjects', default=1, type=int, help='number of subjects in the study')

args = parser.parse_args()

study = args.study_path
nsubjects = args.nsubjects

for sub_id in range(1, nsubjects+1):
    subject = '%s/sub-%.3d_ses-1' % (study,sub_id)

    print('denoising DWI for %s' % subject)

    # remove noise with MRtrix
    os.system('dwidenoise %s/dwi.nii %s/dwi_denoised.nii -noise %s/sigmas.nii -force' % (subject,subject,subject))
