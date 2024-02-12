#!/usr/bin/python
import os,argparse

parser = argparse.ArgumentParser(description='Run DTI and MRDS.')
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

def read_lambdas(lambdas_filename):
    with open(lambdas_filename, 'rt') as lambdas_file:
        line = lambdas_file.readlines()[0]
        lambdas = line.split(' ') [0:3]
        lambdas = [float(value) for value in lambdas]

               #lambda1    #lambda23 
        return lambdas[0], (lambdas[1]+lambdas[2])/2.0

for i in range(nsubjects):
    subject = 'sub-%.3d_ses-1' % (i+1)
    print('├── Subject %s' % subject)

    print('│   ├── Fitting DTI')
    dti_path = '%s/%s/dti' % (study,subject)
    if not os.path.exists( dti_path ):
        os.system('mkdir %s' % (dti_path))
    if args.mask:
        os.system('dti %s/%s/corrected_dwi.nii.gz %s %s/%s/dti/results -mask %s -response 0 -correction 0 > %s/%s/dti/log.txt' % (study,subject,scheme,study,subject,mask,study,subject))
    else:
        os.system('dti %s/%s/corrected_dwi.nii.gz %s %s/%s/dti/results -response 0 -correction 0 > %s/%s/dti/log.txt' % (study,subject,scheme,study,subject,study,subject))

    print('│   ├── Extracting DTI Metrics')
    os.system('tensor2metric %s/%s/dti/results_DTInolin_Tensor.nii.gz \
                        -ad  %s/%s/dti/results_DTInolin_AD.nii.gz     \
                        -rd  %s/%s/dti/results_DTInolin_RD.nii.gz     \
                        -adc %s/%s/dti/results_DTInolin_MD.nii.gz     \
                        -fa  %s/%s/dti/results_DTInolin_FA.nii.gz -force -quiet' % (study,subject,study,subject,study,subject,study,subject,study,subject))

    print('│   ├── Fitting MRDS')
    mrds_path = '%s/%s/mrds' % (study,subject)
    if not os.path.exists( mrds_path ):
        os.system('mkdir %s' % (mrds_path))
    lambda1, lambda23 = read_lambdas('%s/%s/dti/results_DTInolin_ResponseAnisotropic.txt' % (study,subject))
    if args.mask:
        os.system('mdtmrds %s/%s/corrected_dwi.nii.gz %s %s/%s/mrds/results -correction 0 -response %.9f,%.9f -mask %s -modsel %s -each -intermediate -iso -mse -method diff > %s/%s/mrds/log.txt' % (study,subject,scheme,study,subject,lambda1,lambda23,mask,modsel,study,subject))
    else:
        os.system('mdtmrds %s/%s/corrected_dwi.nii.gz %s %s/%s/mrds/results -correction 0 -response %.9f,%.9f -modsel %s -each -intermediate -iso -mse -method diff > %s/%s/mrds/log.txt' % (study,subject,scheme,study,subject,lambda1,lambda23,modsel,study,subject))

    print('│   ├── Extracting MRDS Metrics')
    if args.mask:
        os.system('python mrds2metrics.py %s/%s/mrds --prefix results --method Diff --modsel %s --mask %s' % (study,subject,modsel.upper(),mask))
    else:
        os.system('python mrds2metrics.py %s/%s/mrds --prefix results --method Diff --modsel %s' % (study,subject,modsel.upper()))
