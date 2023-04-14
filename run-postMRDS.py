#!/usr/bin/python3
import os,sys

experiment = sys.argv[1]
nsubjects = int(sys.argv[2])

for sub_id in range(1, nsubjects+1):
    subject = 'sub-%.3d_ses-1' % (sub_id)

    print('processing MRDS output for %s/%s' % (experiment,subject))

    # convert MRDS output to tensor images
    print('\trunning mrds2tensors')
    os.system('python3 mrds2tensors.py %s/%s/mrds/results_MRDS_Diff_BIC_EIGENVALUES.nii    \
                                       %s/%s/mrds/results_MRDS_Diff_BIC_COMP_SIZE.nii      \
                                       %s/%s/mrds/results_MRDS_Diff_BIC_PDDs_CARTESIAN.nii \
                                       %s/%s/mrds/results_MRDS_Diff_BIC_TENSOR.nii' % (experiment,subject,experiment,subject,experiment,subject,experiment,subject))

    # convert MRDS output to diffusivities
    print('\trunning mrds2diffusivities')
    os.system('python3 mrds2diffusivities.py %s/%s/mrds' % (experiment,subject))

    # create input folder for tractometry
    print('\tcreating input folder for tractometry')
    os.system('mkdir %s/input_tractometry' % experiment)
    os.system('mkdir %s/input_tractometry/%s' % (experiment,subject))
    os.system('mkdir %s/input_tractometry/%s/metrics' % (experiment,subject))
    os.system('mkdir %s/input_tractometry/%s/bundles' % (experiment,subject))

    # copy files for tractometry
    print('\tcopying input files for tractometry')
    os.system('cp %s/%s/mrds/results_MRDS_Diff_BIC_PDDs_CARTESIAN.nii %s/input_tractometry/%s/pdd.nii'        % (experiment,subject,experiment,subject))
    os.system('cp %s/%s/mrds/results_MRDS_Diff_BIC_AD.nii             %s/input_tractometry/%s/ad.nii'         % (experiment,subject,experiment,subject))
    os.system('cp %s/%s/mrds/results_MRDS_Diff_BIC_RD.nii             %s/input_tractometry/%s/rd.nii'         % (experiment,subject,experiment,subject))
    os.system('cp %s/%s/mrds/results_MRDS_Diff_BIC_FA.nii             %s/input_tractometry/%s/fa.nii'         % (experiment,subject,experiment,subject))
    os.system('cp %s/%s/mrds/results_MRDS_Diff_BIC_MD.nii             %s/input_tractometry/%s/md.nii'         % (experiment,subject,experiment,subject))
    os.system('cp %s/%s/dti/results_DTInolin_FA.nii                   %s/input_tractometry/%s/metrics/fa.nii' % (experiment,subject,experiment,subject))
    os.system('cp %s/%s/dti/results_DTInolin_AD.nii                   %s/input_tractometry/%s/metrics/ad.nii' % (experiment,subject,experiment,subject))
    os.system('cp %s/%s/dti/results_DTInolin_RD.nii                   %s/input_tractometry/%s/metrics/rd.nii' % (experiment,subject,experiment,subject))
    os.system('cp %s/%s/dti/results_DTInolin_MD.nii                   %s/input_tractometry/%s/metrics/md.nii' % (experiment,subject,experiment,subject))

    # compress files
    print('\tcompressing input files for tractometry')
    os.system('gzip %s/input_tractometry/%s/*' % (experiment,subject))
    os.system('gzip %s/input_tractometry/%s/metrics/*' % (experiment,subject))
