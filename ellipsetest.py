import numpy as np
from matplotlib import pyplot as plt

thetao = np.pi/4
phio = 0
zp = 1
r = 1
R = 5
dt = 0.1

N = 400

ep = r/R
ep2 = ep**2
ct = np.cos(thetao)
st = np.sin(thetao)
ct2 = ct**2
dxb = ep * np.sqrt(1 - ep2) / (ct2 - ep2)
dyb = ep / np.sqrt(ct2 - ep2)
xbo = ct*st / (ct2 - ep2)
ybo = 0
alphas = np.linspace(0,2 * np.pi,100)
xps = zp * dxb * np.cos(alphas) + xbo
yps = zp * dyb * np.sin(alphas) + ybo
plt.plot(xps,yps)
plt.gca().set_aspect('equal')

thetamin = np.arctan(xbo-dxb)
thetamax = np.arctan(xbo+dxb)
phimin = np.arctan(ybo-dyb)
phimax = np.arctan(ybo+dyb)

thetas = np.linspace(thetamin,thetamax,N)
phis = np.linspace(phimin,phimax,N)
ts = np.arange(R-r,R+r,dt)

c = R * np.array(
    (
        np.sin(thetao)*np.cos(phio),
        np.sin(thetao)*np.sin(phio),
        np.cos(thetao)
    )
)

cins = []
couts = []
for theta in thetas:    
    for phi in phis:
        n = np.array(
            (
                np.sin(theta)*np.cos(phi),
                np.sin(theta)*np.sin(phi),
                np.cos(theta)
            )
        )
        coord = zp / np.cos(theta) * n[:2]
        inside = False
        for t in ts:
            q = t * n
            if np.linalg.norm(
                t * n - c
            ) < r:   
                inside = True
                break             
        (
            cins if inside else couts
        ).append(coord)
plt.plot(*zip(*cins),'.')
plt.plot(*zip(*couts),'.')
plt.show()