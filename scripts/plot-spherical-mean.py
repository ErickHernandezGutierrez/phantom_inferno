import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib
from utils import load_scheme

matplotlib.rcParams['font.sans-serif'] = "Comic Sans MS"
matplotlib.rcParams['font.family'] = "sans-serif"

x,y,z = 4,8,0

scheme = load_scheme('../../../data/scheme.txt')
shellmeans = nib.load( 'shellmeans.nii' ).get_fdata()
dti = nib.load( 'mrds/results_mrds_Diff_V1_TENSOR_T0.nii' ).get_fdata()
mrds = [nib.load( 'mrds/results_mrds_Diff_BIC_TENSOR_T%d.nii' % t ).get_fdata() for t in range(3)]
compsize = nib.load( 'mrds/results_mrds_Diff_BIC_COMP_SIZE.nii' ).get_fdata()

alphas = compsize[x,y,z, 0:3]

print(alphas)

bvals = [0]
for b in range(100, 1000, 200):
    bvals.append( b )
for b in range(1000, 2000, 200):
    bvals.append( b )
for b in range(2000, 3200, 200):
    bvals.append( b )
print('num. bvals = ', len(bvals))

dirs = {}
for b in bvals:
    dirs[b] = []
for (dx,dy,dz,b) in scheme:
    dir = np.array([ dx,dy,dz ])
    b = int(b)
    dirs[b].append( dir )
print('g:')
for key in dirs:
    print( key, len(dirs[key]) )

signal = []
tensor_signal = []
multi_tensor_signal = []
for i,b in enumerate(bvals):
    signal.append( shellmeans[x,y,z, i] )

    D = np.zeros( (3,3) )
    D[0,0] = dti[x,y,z,0]
    D[1,1] = dti[x,y,z,1]
    D[2,2] = dti[x,y,z,2]
    D[0,1] = D[1,0] = dti[x,y,z,3]
    D[0,2] = D[2,0] = dti[x,y,z,4]
    D[2,1] = D[1,2] = dti[x,y,z,5]

    tensor_mean = 0
    for g in dirs[b]:
        tensor_mean +=  np.exp(-b * g @ D @ g.transpose() )
    tensor_mean /= len(dirs[b])
    tensor_signal.append( tensor_mean )

    D = [np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3))]
    for t in range(3):
        D[t][0,0] = mrds[t][x,y,z,0]
        D[t][1,1] = mrds[t][x,y,z,1]
        D[t][2,2] = mrds[t][x,y,z,2]
        D[t][0,1] = D[t][1,0] = mrds[t][x,y,z,3]
        D[t][0,2] = D[t][2,0] = mrds[t][x,y,z,4]
        D[t][2,1] = D[t][1,2] = mrds[t][x,y,z,5]

    multi_tensor_mean = 0
    for g in dirs[b]:
        for t in range(3):
            multi_tensor_mean += alphas[t] * np.exp(-b * g @ D[t] @ g.transpose() )
    multi_tensor_mean /= len(dirs[b])
    multi_tensor_signal.append( multi_tensor_mean )

plt.rcParams["font.family"] = 'URW Bookman'
font = {'size' : 35}
matplotlib.rc('font', **font)

fig, axs = plt.subplots(nrows=1, ncols=2)
#axs[0].set_facecolor("gray")
axs[0].plot(bvals, signal, linewidth=4, color='#E9002D', label='signal (single fascicle)', linestyle='dashed')
axs[0].plot(bvals, tensor_signal, linewidth=4, color='#FFAA00', label='MTM signal (1 tensor)')
axs[0].plot(bvals, multi_tensor_signal, linewidth=4, color='#00B000', label='MTM signal (3 tensors)')
axs[0].set_xlabel('b-val')
axs[0].set_ylabel(r'$S/S_0$')
axs[0].legend(loc='upper right', fontsize=20)
axs[0].grid(True)
axs[0].set_yticks(np.arange(0.2, 1.1, 0.1))

#axs[1].set_facecolor("gray")
axs[1].plot(bvals, np.log(signal), linewidth=4, color='#E9002D', label='signal (single fascicle)', linestyle='dashed')
axs[1].plot(bvals, np.log(tensor_signal), linewidth=4, color='#FFAA00', label='MTM signal (1 tensor)')
axs[1].plot(bvals, np.log(multi_tensor_signal), linewidth=4, color='#00B000', label='MTM signal (3 tensors)')
axs[1].set_xlabel('b-val')
axs[1].set_ylabel(r'$log(S/S_0)$')
axs[1].legend(loc='upper right', fontsize=20)
axs[1].grid(True)

plt.show()
