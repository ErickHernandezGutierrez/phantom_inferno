#!/usr/bin/python
import os,sys,argparse

parser = argparse.ArgumentParser(description='Denoise phantom signal.')
parser.add_argument('--study_path', help='path to the study folder')
parser.add_argument('--nsubjects', default=1, type=int, help='number of subjects in the study')
parser.add_argument('--scheme', help='protocol file in format Nx4 text file with [X Y Z b] at each row')
parser.add_argument('--mask', help='mask filename')

args = parser.parse_args()

study = args.study_path
nsubjects = args.nsubjects
scheme = args.scheme
mask = args.mask

for sub_id in range(1, nsubjects+1):
    subject = '%s/sub-%.3d_ses-1' % (study,sub_id)

    print('running DTI for %s' % subject)

    # fit DTI
    os.system('mkdir %s/dti' % subject)
    os.system('dti %s/dwi_corrected.nii %s %s/dti/results.nii -mask %s -response 0 -correction 0' % (subject,scheme,subject,mask))

    # extract DTI metrics
    os.system('tensor2metric %s/dti/results_DTInolin_Tensor.nii \
                        -ad  %s/dti/results_DTInolin_AD.nii     \
                        -rd  %s/dti/results_DTInolin_RD.nii     \
                        -adc %s/dti/results_DTInolin_MD.nii     \
                        -fa  %s/dti/results_DTInolin_FA.nii -force' % (subject,subject,subject,subject,subject))
