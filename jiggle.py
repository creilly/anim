import numpy as np

jiggle_gain = 0.4

# generates a (n X shape) array of randomly forced harmonic motion
# n = 200 at default params `a` (spring constant), `b` (damping), `c` (force amplitude) generates an appealing motion
a = 400
b = 4
c = 40
def generate_jiggles(n,shape,gain=jiggle_gain,a=a,b=b,c=c):
    m = 100
    xo = np.zeros(shape)
    vo = np.zeros(shape)
    fs = np.random.uniform(-gain,gain,(n,*shape))
    xs = []
    nn = 0
    while nn < n:
        x = xo + vo / m
        v = vo + ((-a*xo) + (-b*vo) + c*fs[nn])/m
        xs.append(xo)
        xo = x
        vo = v
        nn += 1
    return xs