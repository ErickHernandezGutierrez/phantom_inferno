import numpy as np
import matplotlib.pyplot as plt
import matplotlib,pickle

# Example dataset
x = np.array([
0.25,
0.3,
0.35,
0.4,
0.45,
0.5,
0.55,
0.6,
0.65,
0.7,
0.75,
0.8,
0.85,
0.9,
0.95,
1
])  # x values

x_demyelination = np.array([0.25, 0.25628617, 0.26763765, 0.27671433, 0.29968126, 0.33839779,
 0.39423744, 0.45766575, 0.51574586, 0.5769337,  0.64969518, 0.73803655,
 0.81022099, 0.87762431, 0.9264393,  1.        ])

y_demyelination = np.array([0.375, 0.42672535, 0.47580492, 0.52301349, 0.57086698, 0.61705801,
 0.6614416,  0.70502417, 0.74917127, 0.79299033, 0.83558773, 0.87654059,
 0.9191989,  0.96236188, 1.00748696, 1.05      ])

y_demyelination = np.array([0.375, 0.40672535, 0.45580492, 0.47301349, 0.5086698, 0.57705801,
 0.6614416,  0.70502417, 0.74917127, 0.79299033, 0.83558773, 0.87654059,
 0.9191989,  0.96236188, 1.00748696, 1.05      ])

y_normal = np.array([
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

"""
y_demyelination = np.array([
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
])  # original y values
"""

y_axonloss = np.array([
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

def fit_polynomial(x, y, coefs_filename, points_filename, degree=5):
    points = np.zeros((len(x),2))
    points[:,0] = np.array(x)
    points[:,1] = np.array(y)
    np.save(points_filename, points)

    # Ensure the polynomial passes through the first and last points by subtracting the baseline
    baseline_y = np.poly1d(np.polyfit([x[0], x[-1]], [y[0], y[-1]], 1))(x)

    # Fit the polynomial to the adjusted y values
    adjusted_y = y - baseline_y
    coefficients = np.polyfit(x, adjusted_y, degree)
    np.save(coefs_filename, coefficients)

    # Create the polynomial function
    polynomial_function = np.poly1d(coefficients)

    # Adjust the polynomial output to add back the baseline
    final_polynomial = lambda x_val: polynomial_function(x_val) + np.poly1d(np.polyfit([x[0], x[-1]], [y[0], y[-1]], 1))(x_val)

    return final_polynomial

# Fit Polynomials
p_normal = fit_polynomial(x, y_normal, 'coefs_normal.npy', 'points_normal.npy')
p_demyelination = fit_polynomial(x_demyelination, y_demyelination, 'coefs_demyelination.npy', 'points_demyelination.npy')
p_axonloss = fit_polynomial(x, y_axonloss, 'coefs_axonloss.npy', 'points_axonloss.npy')

# Plotting
font_size=12
font = {'size' : font_size}
matplotlib.rc('font', **font)
x_range = np.linspace(min(x), max(x), 100)
x_demyelination_range = np.linspace(min(x_demyelination), max(x_demyelination), 100)
plt.scatter(x, y_normal, label='Points Normal', color='black', marker='o')
plt.plot(x_range, p_normal(x_range), label='Polynomial Normal', color='black')

plt.scatter(x_demyelination, y_demyelination, label='Points Demyelination', color='green', marker='x')
plt.plot(x_demyelination_range, p_demyelination(x_demyelination_range), label='Polynomial Demyelination', color='lightgreen')

plt.scatter(x, y_axonloss, label='Points Axon Loss', color='blue', marker='^')
plt.plot(x_range, p_axonloss(x_range), label='Polynomial Axon Loss', color='lightblue')

plt.legend()
plt.show()
