import argparse, os

parser = argparse.ArgumentParser(description='Generate phantom signal.')
parser.add_argument('phantom', help='path to the phantom template')
parser.add_argument('study_path', help='path to the study folder')
parser.add_argument('scheme', help='protocol file in format Nx4 text file with [X Y Z b] at each row')
parser.add_argument('--nsubjects',  default=1,    type=int, help='number of subjects in the study. default: 1')
args = parser.parse_args()

phantom_info = {
    'templates/Training_3D_SF': {'nbundles': 3, 'dims': [16,16,5]},
    'templates/Training_SF': {'nbundles': 5, 'dims': [16,16,5]},
    'templates/Phantomas': {'nbundles': 20, 'dims': [50,50,50]}
}

phantom = args.phantom
study = args.study_path
scheme = args.scheme
nsubjects = args.nsubjects
nbundles = phantom_info[phantom]['nbundles']

for i in range(nsubjects):
    subject = 'sub-%.3d_ses-1' % (i+1)

    for bundle in range(1,nbundles+1):
        os.system('dwi2tensor %s/ground_truth/%s/bundle-%d__dwi.nii.gz %s/ground_truth/%s/bundle-%d__tensor.nii.gz \
                        -grad %s \
                        -mask %s/bundle-%d__wm-mask.nii.gz \
                        -force' % (study,subject,bundle,study,subject,bundle,scheme,phantom,bundle))

        os.system('tensor2metric  %s/ground_truth/%s/bundle-%d__tensor.nii.gz \
                             -fa  %s/ground_truth/%s/bundle-%d__fa.nii.gz     \
                             -rd  %s/ground_truth/%s/bundle-%d__rd.nii.gz     \
                             -ad  %s/ground_truth/%s/bundle-%d__ad.nii.gz     \
                             -adc %s/ground_truth/%s/bundle-%d__md.nii.gz     \
                             -force' % (study,subject,bundle,study,subject,bundle,study,subject,bundle,study,subject,bundle,study,subject,bundle,))
