import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

x = np.arange(0.25,1.05,0.05)

d_perp_normal = np.array([
0.375,
0.4375,
0.488372093,
0.525,
0.5675675676,
0.6,
0.652173913,
0.6774193548,
0.724137931,
0.75,
0.8076923077,
0.867768595,
0.8974358974,
0.9545454545,
1,
1.05
])

d_perp_demyeliation = np.array([
0.375,
0.5384615385,
0.5614973262,
0.6,
0.65625,
0.6687898089,
0.724137931,
0.75,
0.7664233577,
0.8076923077,
0.84,
0.875,
0.9130434783,
0.9545454545,
1,
1.05
])

d_perp_axonloss = np.array([
0.375,
0.3559322034,
0.3620689655,
0.3860294118,
0.4038461538,
0.4375,
0.488372093,
0.546875,
0.6,
0.65625,
0.724137931,
0.8076923077,
0.875,
0.9375,
0.9813084112,
1.05
])

def tortuosity_normal(x):
    a = -0.2666563
    b = 4.359128
    c = -9.797455
    d = 12.14556
    e = -6.677178
    f = 1.285754
    return a + b*x + c*x*x + d*x*x*x + e*x*x*x*x + f*x*x*x*x*x

def tortuosity_demyelination(x):
    a = -1.94486
    b = 19.75048
    c = -61.38125
    d = 95.6805
    e = -72.65727
    f = 21.60637
    return a + b*x + c*x*x + d*x*x*x + e*x*x*x*x + f*x*x*x*x*x

def tortuosity_axonloss(x):
    a = 0.7249711
    b = -2.67498
    c = 6.257764
    d = -5.676736
    e = 3.976406
    f = -1.562527
    return a + b*x + c*x*x + d*x*x*x + e*x*x*x*x + f*x*x*x*x*x

d_par = 2
d_perp_sano = x
a = 0.0
b = 12.45163
c = -45.29592
d = 82.64623
e = -71.37607
f = 23.50357
#d_perp_demyeliation = a + b*x + c*x*x + d*x*x*x + e*x*x*x*x + f*x*x*x*x*x 

plt.plot(x,d_perp_normal, color='black',label='identity')
plt.plot(x,d_perp_demyeliation,color='green',label='demyelination')
plt.plot(x,d_perp_axonloss,color='blue',label='axon loss')
#plt.plot(x,tortuosity_normal(x), color='black',label='identity')
#plt.plot(x,tortuosity_demyelination(x),color='green',label='demyelination')
#plt.plot(x,tortuosity_axonloss(x),color='blue',label='axon loss')
plt.ylabel(r'$D^\perp$')
plt.xlabel(r'$1-f$')
plt.xticks()
plt.grid(True)
plt.legend(loc='upper left')
plt.show()
