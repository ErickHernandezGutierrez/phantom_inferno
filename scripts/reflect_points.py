import numpy as np
import matplotlib.pyplot as plt

def d(x1,y1):
    return np.abs(-0.9*x1 + y1 - 0.15) / np.sqrt((0.9**2)+1)

def xr(x1,y1):
    return (y1-d(x1,y1)-0.15) / 0.9

def yr(x1,y1):
    return 0.9*xr(x1,y1) + 0.15

points_x = np.array([
0.25,
0.30,
0.35,
0.40,
0.45,
0.50,
0.55,
0.60,
0.65,
0.70,
0.75,
0.80,
0.85,
0.90,
0.95,
1.00,
])
points_y = np.array([
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
1.05,
])

reflected_points_x = []
reflected_points_y = []

L_points_x = []
L_points_y = []

m = 0.9
m_perp = -10.0/9.0
t = 0.15

for (xP,yP) in zip(points_x,points_y):
    t_perp = yP - m_perp*xP

    xL = (t_perp-t) / (m-m_perp)
    yL = m*xL + t

    L_points_x.append(xL)
    L_points_y.append(yL)

    dx = np.abs(xL-xP)
    dy = np.abs(yL-yP)

    x = xL-dx
    y = dy+yL

    reflected_points_x.append(x)
    reflected_points_y.append(y)

L_points_x = np.array(L_points_x)
L_points_y = np.array(L_points_y)

reflected_points_x = np.array(reflected_points_x)
reflected_points_y = np.array(reflected_points_y)

print(reflected_points_x)
print(reflected_points_y)

plt.plot([0.25,1], [0.375,1.05],color='black')
plt.scatter(reflected_points_x, reflected_points_y, label='Points Demyelination', color='green', marker='x')
plt.scatter(L_points_x, L_points_y, label='Points L', color='yellow', marker='o')
plt.scatter(points_x, points_y, label='Points Axon Loss', color='blue', marker='^')
plt.legend()
plt.show()
