import numpy as np
import nibabel as nib
import sys
from utils import success_rate

ref_filename     = sys.argv[1]
mosemap_filename = sys.argv[2]
#mask_filename = sys.argv[3]

ref = nib.load( ref_filename ).get_fdata().astype(np.int32)
mosemap = nib.load( mosemap_filename ).get_fdata().astype(np.int32)
#mask = nib.load( mask_filename ).get_fdata().astype(np.int32)

sr = success_rate(ref, mosemap)

print('success = %f' % (sr))
