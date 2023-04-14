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

def read_lambdas(lambdas_filename):
    with open(lambdas_filename, 'rt') as lambdas_file:
        line = lambdas_file.readlines()[0]
        lambdas = line.split(' ') [0:3]
        lambdas = [float(value) for value in lambdas]

               #lambda1    #lambda23 
        return lambdas[0], (lambdas[1]+lambdas[2])/2.0

for sub_id in range(1, nsubjects+1):
    subject = '%s/sub-%.3d_ses-1' % (study,sub_id)

    print('running MRDS for %s' % subject)

    # get lambdas from DTI
    lambda1, lambda23 = read_lambdas('%s/dti/results_DTInolin_ResponseAnisotropic.txt' % subject)

    # fit MRDS
    os.system('mkdir %s/mrds' % subject)
    os.system('mdtmrds %s/dwi_corrected.nii %s %s/mrds/results.nii -correction 0 -response %.9f,%.9f -mask %s -modsel bic -each -intermediate -fa -md -mse -method diff 1' % (subject,scheme,subject,lambda1,lambda23,mask))

    os.system('python3 mrds2tensors.py %s/mrds/results_MRDS_Diff_BIC_EIGENVALUES.nii    \
                                       %s/mrds/results_MRDS_Diff_BIC_COMP_SIZE.nii      \
                                       %s/mrds/results_MRDS_Diff_BIC_PDDs_CARTESIAN.nii \
                                       %s/mrds/results_MRDS_Diff_BIC_TENSOR.nii' % (subject,subject,subject,subject))

    os.system('python3 mrds2diffusivities.py %s/mrds' % (subject))
