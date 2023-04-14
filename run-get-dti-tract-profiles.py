#!/usr/bin/python3
import os,sys

experiment = sys.argv[1]

for sub_id in range(1, 27):
    subject = 'sub-%.3d_ses-1' % (sub_id)

    print('\ngetting DTI metrics for %s/' % subject)

    for bundle in range(1,4):
        for metric in ['AD', 'RD', 'FA', 'MD']:
            os.system('mrcalc %s/%s/dti/results_DTInolin_%s.nii gt/mask-%d.nii -multiply %s/results_tractometry/%s/bundle-%d_tract-%s-profile.nii -force' % (experiment,subject,metric, bundle, experiment,subject, bundle,metric))

