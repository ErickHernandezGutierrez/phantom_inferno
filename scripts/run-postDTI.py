#!/usr/bin/python
import os,sys,argparse

parser = argparse.ArgumentParser(description='Denoise phantom signal.')
parser.add_argument('--study_path', help='path to the study folder')
parser.add_argument('--nsubjects', default=1, type=int, help='number of subjects in the study')

args = parser.parse_args()

study = args.study_path
nsubjects = args.nsubjects

for sub_id in range(1, nsubjects+1):
    subject = 'sub-%.3d_ses-1' % (sub_id)

    print('processing DTI output for %s/%s' % (study,subject))

    # convert DTI output to  diffusivities
    os.system('tensor2metric  %s/%s/dti/results_DTInolin_Tensor.nii \
                        -fa   %s/%s/dti/results_DTInolin_FA.nii     \
                        -ad   %s/%s/dti/results_DTInolin_AD.nii     \
                        -rd   %s/%s/dti/results_DTInolin_RD.nii     \
                        -adc  %s/%s/dti/results_DTInolin_MD.nii -force' % (study,subject,study,subject,study,subject,study,subject,study,subject))
