import nibabel as nib
from nibabel.streamlines import Field
from nibabel.orientations import aff2axcodes
import sys, argparse

parser = argparse.ArgumentParser(description='Convert .tck to .trk or viceversa.')
parser.add_argument('input_file',  help='input tractogram file')
parser.add_argument('output_file', help='output tractogram file')
parser.add_argument('--reference', help='reference anatomy for tck/trk file')
args = parser.parse_args()

input_filename = args.input_file
output_filename = args.output_file
anatomy_filename = args.reference

anatomy_file = nib.load(anatomy_filename)

header = {}
header[Field.VOXEL_TO_RASMM] = anatomy_file.affine.copy()
header[Field.VOXEL_SIZES] = anatomy_file.header.get_zooms()[:3]
header[Field.DIMENSIONS] = anatomy_file.shape[:3]
header[Field.VOXEL_ORDER] = "".join(aff2axcodes(anatomy_file.affine))

input_file = nib.streamlines.load(input_filename)
nib.streamlines.save(input_file.tractogram, output_filename, header=header)
