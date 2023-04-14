#!/usr/bin/python3
import os, sys

experiment = sys.argv[1]
nsubjects = int(sys.argv[2])

for sub_id in range(1, nsubjects):
    subject = 'sub-%.3d_ses-1' % sub_id

    print('generating DWI for %s/%s' % (experiment,subject))

    # remove noise
    os.system('python3 phantomgen.py %s %s' % (experiment,subject))
