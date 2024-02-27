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

print('\n│Pipeline - Extract Metrics│')

input_tractometry = '%s/input_tractometry' % (study)
if not os.path.exists( input_tractometry ):
    os.system('mkdir %s' % (input_tractometry))

for i in range(nsubjects):
    subject = 'sub-%.3d_ses-1' % (i+1)
    print('├── Subject %s' % subject)

    print('│   ├── Extracting DTI Metrics')
    os.system('tensor2metric %s/%s/dti/results_DTInolin_Tensor.nii.gz \
                        -ad  %s/%s/dti/results_DTInolin_AD.nii.gz     \
                        -rd  %s/%s/dti/results_DTInolin_RD.nii.gz     \
                        -adc %s/%s/dti/results_DTInolin_MD.nii.gz     \
                        -fa  %s/%s/dti/results_DTInolin_FA.nii.gz -force -quiet' % (study,subject,study,subject,study,subject,study,subject,study,subject))

    # create tractometry input folders
    if not os.path.exists( '%s/%s' % (input_tractometry,subject) ):
        os.system('mkdir %s/%s' % (input_tractometry,subject))
    if not os.path.exists( '%s/%s/metrics' % (input_tractometry,subject) ):
        os.system('mkdir %s/%s/metrics' % (input_tractometry,subject))
    if not os.path.exists( '%s/%s/fixel_metrics' % (input_tractometry,subject) ):
        os.system('mkdir %s/%s/fixel_metrics' % (input_tractometry,subject))

    # copy data to input tractometry folders
    os.system('cp %s/%s/dti/results_DTInolin_FA.nii.gz %s/%s/metrics/fa.nii.gz' % (study,subject,input_tractometry,subject))
    os.system('cp %s/%s/dti/results_DTInolin_MD.nii.gz %s/%s/metrics/md.nii.gz' % (study,subject,input_tractometry,subject))
    os.system('cp %s/%s/dti/results_DTInolin_RD.nii.gz %s/%s/metrics/rd.nii.gz' % (study,subject,input_tractometry,subject))
    os.system('cp %s/%s/dti/results_DTInolin_AD.nii.gz %s/%s/metrics/ad.nii.gz' % (study,subject,input_tractometry,subject))
    os.system('cp %s/%s/mrds/results_MRDS_Diff_%s_FA.nii.gz %s/%s/fixel_metrics/fixel_fa.nii.gz' % (study,subject,modsel.upper(),input_tractometry,subject))
    os.system('cp %s/%s/mrds/results_MRDS_Diff_%s_MD.nii.gz %s/%s/fixel_metrics/fixel_md.nii.gz' % (study,subject,modsel.upper(),input_tractometry,subject))
    os.system('cp %s/%s/mrds/results_MRDS_Diff_%s_RD.nii.gz %s/%s/fixel_metrics/fixel_rd.nii.gz' % (study,subject,modsel.upper(),input_tractometry,subject))
    os.system('cp %s/%s/mrds/results_MRDS_Diff_%s_AD.nii.gz %s/%s/fixel_metrics/fixel_ad.nii.gz' % (study,subject,modsel.upper(),input_tractometry,subject))
    os.system('cp %s/%s/mrds/results_MRDS_Diff_%s_PDDs_CARTESIAN.nii.gz %s/%s/pdds.nii.gz' % (study,subject,modsel.upper(),input_tractometry,subject))
