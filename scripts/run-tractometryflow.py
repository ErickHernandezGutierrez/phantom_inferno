#!/usr/bin/python3
import os,sys,argparse

"""
parser = argparse.ArgumentParser(description='Denoise phantom signal.')
parser.add_argument('--study_path', help='path to the study folder')
parser.add_argument('--nsubjects', default=1, type=int, help='number of subjects in the study')

args = parser.parse_args()

study = args.study_path
nsubjects = args.nsubjects

for sub_id in range(1, nsubjects+1):
    subject = 'sub-%.3d_ses-1' % (sub_id)

    # create input folder for tractometry
    print('\tcreating input folder for tractometry')
    os.system('mkdir %s/input_tractometry' % study)
    os.system('mkdir %s/input_tractometry/%s' % (study,subject))
    os.system('mkdir %s/input_tractometry/%s/metrics' % (study,subject))
    os.system('mkdir %s/input_tractometry/%s/bundles' % (study,subject))

    # copy files for tractometry
    print('\tcopying input files for tractometry')
    os.system('cp %s/%s/mrds/results_MRDS_Diff_BIC_PDDs_CARTESIAN.nii %s/input_tractometry/%s/pdd.nii'        % (study,subject,study,subject))
    os.system('cp %s/%s/mrds/results_MRDS_Diff_BIC_AD.nii             %s/input_tractometry/%s/ad.nii'         % (study,subject,study,subject))
    os.system('cp %s/%s/mrds/results_MRDS_Diff_BIC_RD.nii             %s/input_tractometry/%s/rd.nii'         % (study,subject,study,subject))
    os.system('cp %s/%s/mrds/results_MRDS_Diff_BIC_FA.nii             %s/input_tractometry/%s/fa.nii'         % (study,subject,study,subject))
    os.system('cp %s/%s/mrds/results_MRDS_Diff_BIC_MD.nii             %s/input_tractometry/%s/md.nii'         % (study,subject,study,subject))
    os.system('cp %s/%s/dti/results_DTInolin_FA.nii                   %s/input_tractometry/%s/metrics/fa.nii' % (study,subject,study,subject))
    os.system('cp %s/%s/dti/results_DTInolin_AD.nii                   %s/input_tractometry/%s/metrics/ad.nii' % (study,subject,study,subject))
    os.system('cp %s/%s/dti/results_DTInolin_RD.nii                   %s/input_tractometry/%s/metrics/rd.nii' % (study,subject,study,subject))
    os.system('cp %s/%s/dti/results_DTInolin_MD.nii                   %s/input_tractometry/%s/metrics/md.nii' % (study,subject,study,subject))

    # compress files
    print('\tcompressing input files for tractometry')
    os.system('gzip %s/input_tractometry/%s/*' % (study,subject))
    os.system('gzip %s/input_tractometry/%s/metrics/*' % (study,subject))
"""

os.system('../../nextflow/nextflow ../../tractometry_flow/main.nf --input input_tractometry --use_provided_centroids false --mean_std_per_point_density_weighting true --mean_std_density_weighting true -resume')