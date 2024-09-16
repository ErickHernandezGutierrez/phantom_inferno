#!/usr/bin/python
import os,argparse

parser = argparse.ArgumentParser(description='Track with iFOD2 using multi-tensor ODFs.')
parser.add_argument('study_path', help='path to the study folder')
parser.add_argument('scheme', help='protocol file in format Nx4 text file with [X Y Z b] at each row')
parser.add_argument('--modsel', default='bic', help='model selector (aic,bic,ftest). default: bic')
parser.add_argument('--mask', help='mask filename')
parser.add_argument('--nsubjects', default=1, type=int, help='number of subjects in the study. default: 1')
parser.add_argument('--ntracks', default=100000, type=int, help='number of selected streamlines for the tractogram. default: 100000')
parser.add_argument('--roi', help='ROIs filename')
args = parser.parse_args()

study = args.study_path
scheme = args.scheme
modsel = args.modsel
mask = args.mask
nsubjects = args.nsubjects
ntracks = args.ntracks
roi = args.roi

print('│Pipeline - Step 4 (Fiber Tracking & Tractogram Segmentation)│')

for i in range(nsubjects):
    subject = 'sub-%.3d_ses-1' % (i+1)
    input_tractometry = '%s/input_tractometry/%s' % (study,subject)
    print('├── Subject %s' % subject)

    # create tractometry input folders
    if not os.path.exists( '%s/bundles'%(input_tractometry) ):
        os.system( 'mkdir %s/bundles'%(input_tractometry) )
    if not os.path.exists( '%s/%s/mrds/results_MRDS_Diff_%s_BUNDLES'%(study,subject,modsel.upper()) ):
        os.system( 'mkdir %s/%s/mrds/results_MRDS_Diff_%s_BUNDLES'%(study,subject,modsel.upper()) )

    print('│   ├── Fiber tracking')
    os.system('tckgen %s/%s/mrds/results_MRDS_Diff_%s_ODF_SH.nii.gz %s/%s/mrds/results_MRDS_Diff_%s_TRACKS.tck -select %d -seed_image %s -grad %s -force' % (study,subject,modsel.upper(),study,subject,modsel.upper(),ntracks,mask,scheme))

    print('│   ├── Tractogram segmentation')
    os.system('tck2connectome %s/%s/mrds/results_MRDS_Diff_%s_TRACKS.tck %s %s/%s/mrds/results_MRDS_Diff_%s_CONNECTOME.csv -assignment_radial_search 1.0 -out_assignments %s/%s/mrds/results_MRDS_Diff_%s_NODES.txt -force' % (study,subject,modsel.upper(),roi,study,subject,modsel.upper(),study,subject,modsel.upper()))
    os.system('connectome2tck %s/%s/mrds/results_MRDS_Diff_%s_TRACKS.tck %s/%s/mrds/results_MRDS_Diff_%s_NODES.txt %s/%s/mrds/results_MRDS_Diff_%s_BUNDLES/bundle -files per_edge -force' % (study,subject,modsel.upper(),study,subject,modsel.upper(),study,subject,modsel.upper()))

    # copy data to tractometry input folder
    os.system('for roi in 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39; do \
                    cp %s/%s/mrds/results_MRDS_Diff_%s_BUNDLES/bundle$roi-$(($roi+1)).tck %s/bundles/bundle-$(((($roi+1))/2)).tck; \
               done'%(study,subject,modsel.upper(),input_tractometry))
    os.system('for j in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do \
                    python scripts/convert_tractogram.py %s/bundles/bundle-${j}.tck %s/bundles/bundle-${j}.trk --reference %s/%s/dwi.nii.gz; \
               done'%(input_tractometry,input_tractometry,study,subject))
    os.system('rm %s/bundles/bundle*.tck'%(input_tractometry))
