import numpy as np

c = np.cos
s = np.sin
# camera coord axes is obtained from reference coord axes 
# aligned with "lab-frame" axis by the following sequence of rotations:
# 1. rotation by angle t1 about z-axis ('roll')
# 2. rotation by angle t2 about x-axis ('tilt')
# 3. rotation by angle t3 about y-axis ('azim')
def get_pixel_matrix(t1,t2,t3):
    (c1,s1),(c2,s2),(c3,s3) = (
        (
            np.cos(t),np.sin(t)
        ) for t in (t1,t2,t3)
    )
    return np.array(
        (
            (
                 c1*c3 + s1*s2*s3,  s1*c2, -c1*s3 + s1*s2*c3
            ),(
                -s1*c3 + c1*s2*s3,  c1*c2,  s1*s3 + c1*s2*c3
            ),(
                 c2*s3           , -s2   ,  c2*c3
            )
        )
    )
zop = 1
def get_pixel(m,p,q):
    xbar, ybar, zbar = m.dot(q-p)
    t = zbar / zop
    return xbar / t, ybar / t, zbar

# m : transformation matrix
# p : pin hole location
# q : sphere center
# r : sphere radius
def get_ellipse(m,p,q,r):    
    qp = q - p
    R = np.linalg.norm(qp)    
    dx, dy, dz = m.dot(qp)
    dxpar = np.array((dx,dy))    
    dxparnorm = np.linalg.norm(dxpar)    
    th = np.arctan(abs(dxparnorm/dz))
    ep = r / R
    c = np.cos(th)
    s = np.sin(th)
    c2 = c**2
    ep2 = ep**2
    xbo = c * s / (c2 - ep2)
    ybo = 0
    dxb = ep * np.sqrt(1 - ep2) / (c2 - ep2)
    dyb = ep / np.sqrt(c2 - ep2)    
    if dxparnorm > 0:
        xbhat = dxpar / dxparnorm
    else:
        xbhat = np.array([1,0])
    ybhat = np.cross([0,0,1],[*xbhat,0])[:2]
    return (
        (xbhat,xbo,dxb),
        (ybhat,ybo,dyb)
    )

# roll-free camera orientation
# to look at point q from p
def point_camera(p,q):
    qpx, qpy, qpz = q - p    
    theta = -np.arcsin(
        qpy/np.sqrt(qpx**2 + qpy**2 + qpz**2)
    )
    phi = +np.arctan2(qpx,qpz)
    return 0.0, theta, phi

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import matplotlib.animation as an
    nframes = 100
    r = 10    
    phis = 2 * np.pi * np.arange(nframes) / nframes
    theta = np.pi/4
    def get_p(r,m):
        return -r * m.transpose().dot([0,0,1])
    gvecs = [
        [ 3,-2, 0],
        [ 0,-2, 3],
        [-3,-2, 0],
        [ 0,-2,-3],
        [ 3,-2, 0]
    ]
    cvecs = (
        ( 0, 0, 0)
    )
    hvecs = (
        ( 1, 1, 1),
        (-1,-1, 1),
        ( 1,-1,-1),
        (-1, 1,-1)
    )    
    glines = [
        plt.plot([],[],color='black')[0]    
        for gvec in gvecs
    ]
    hlines = [
        plt.plot([],[],'o',ms=30,color='black',mfc='red')[0]
        for hvec in hvecs
    ]
    clines = [
        plt.plot([],[],'o',ms=30,color='black',mfc='blue')[0]     
        for cvec in cvecs
    ]    
    def update(frame):        
        t1 = 0.0
        t2 = theta
        t3 = phis[frame]
        m = get_pixel_matrix(t1,t2,t3)
        p = get_p(r,m)
        toplot = []
        for isg, vecs, lines in (
            (True , gvecs, glines),
            (False, cvecs, clines),
            (False, hvecs, hlines)
        ):
            gpoints = []           
            for vec, line in zip(vecs,lines):        
                xp, yp, zp = get_pixel(
                    m, p, np.array(vec)
                )
                if isg:
                    gpoints.append((xp,yp))
                else:
                    toplot.append(
                        (zp,(xp,yp),line)
                    )
            if isg:
                xdata, ydata = zip(*gpoints)                
                line.set_xdata(xdata)
                line.set_ydata(ydata)
                line.set_zorder(-1)
        else:
            for zorder, (_, (xp,yp), line) in enumerate(
                sorted(toplot,reverse=True)
            ):
                line.set_xdata([xp])
                line.set_ydata([yp])
                line.set_zorder(zorder)
        return (*hlines,*clines,*glines)
    limscale = ls = 0.25
    plt.xlim(-ls,ls)
    plt.ylim(-ls,ls)
    plt.gca().set_aspect('equal')
    ani = an.FuncAnimation(plt.gcf(),func=update,frames=nframes,interval=30)    
    plt.show()