import numpy as np, svgtools as svgt, util, perspective as pers, methane, animate as anim
from PIL import Image
from matplotlib import pyplot as plt

deltat = 1
power = 2
wiggles = 1.5
amp = 1
sigmas = 2.5
nt = 500

def plot_pulse(
    canvas_center, scale, 
    c, m, p,     
    v, w, amp, to,
    sc, st, so    
):
    qs = generate_trajectory(c,v,amp*w,to)
    return plot_trajectory(
        canvas_center, scale, 
        m, p, qs, 
        sc, st, so        
    )

def plot_trajectory(
        canvas_center, scale, 
        m, p, qs, 
        sc, st, so
    ):
    qps = qs - np.outer(np.ones(qs.shape[0]),p)
    qppts = np.einsum('ik,jk',m,qps)
    dpart = qppts[:2]
    dperpt = qppts[2]
    nodes = (dpart / dperpt).transpose()*scale+canvas_center
    return svgt.generate_wavy(
        nodes,
        st,sc,so,
        buffer=0.7,search_length=20
    )  

def split_pulse(qs,obstacles):
    collisions = []
    for qindex, q in enumerate(qs):
        for c, r in obstacles:
            if np.linalg.norm(q-c) < r:
                collisions.append(qindex)
                break
    if len(collisions) == 0:
        segments = [qs]
    else:
        segments = [qs[:min(collisions)], qs[max(collisions)+1:]]
    return [s for s in segments if len(s) > (3*20+1)]

# given specified start and stop, generate 
# array of pulse center coords
def get_pulse_coords(cstart, cend, n):
    dc = cend - cstart    
    vhat = dc / np.linalg.norm(dc)
    return np.outer(
        np.linspace(0,1,n), 
        dc
    ) + np.outer(
        np.ones(n), 
        cstart
    ), vhat

def generate_trajectory(c,v,w,to):
    ts = sigmas * deltat * np.linspace(-1,1,nt)
    return sum(
        np.outer(tarr,varr)
        for tarr, varr in (
            (np.ones(ts.shape),c),(ts,v),(pulsef(ts,to,deltat,power,wiggles,amp),w)
        )
    )
    
def pulsef(t,to,deltat,power,wiggles,amp):
    return amp * np.exp(-1/2*(t/deltat)**power) * np.cos(
        2 * np.pi * (t - to) * wiggles / deltat
    )

if __name__ == '__main__':

    folder = 'pims'
    width, height = 1280, 720

    sc = '#842dcc'
    st = 12
    so = 1.0
    scale = 500
    frames = 50

    vmin = -5
    vmax = +10

    amp = 1.0

    oscillations = 25

    v = np.array([-1,0,1])
    vhat = v / np.linalg.norm(v)
    w = np.array([0,1,0])    
    rbuffer = 6

    pphi = +np.pi/6
    ptheta = -np.pi/9

    sphere_center = qo = np.zeros(3)
    r_sphere = 1

    cmin, cmax = [qo + cv * v for cv in (vmin,vmax)]

    pulse_centers, vhat = get_pulse_coords(cmin,cmax,frames)

    qclose = cmin - sigmas * deltat * v    

    a1, a2, a3 = 0.0, ptheta, pphi
    m = pers.get_pixel_matrix(a1,a2,a3)
    # invert yp coord so my head stops hurting
    m = np.array([[1,0,0],[0,-1,0],[0,0,1]]).dot(m)        

    r = -m.dot(qclose)[2] + rbuffer
    p = r * m.transpose().dot([0,0,-1])      

    canvas_center = np.array((width/2,height/2))    

    deltatw = deltat / wiggles
    tos = -np.arange(frames)/frames * oscillations * deltatw    
    print('precleaning')
    for ext in ('png','svg'):
        util.clean_folder(folder,ext)
    print('generating svgs')
    for frame in range(frames):    
        pulse_center = pulse_centers[frame]
        to = tos[frame] 
        traj = generate_trajectory(pulse_center, v, w, to)
        segs = split_pulse(traj,[(qo,r_sphere)])
        dzs = [
            np.average(
                np.einsum(
                    'ij,nj->in',m,s - np.outer(
                        np.ones(s.shape[0]),p
                    )
                )[2]
            )  for s in segs
        ]        
        els = [
            (
                dz, plot_trajectory(
                    canvas_center, scale, m, p, s,  
                    sc, st, so
                ) 
            ) for dz, s in zip(dzs, segs)
        ]             
        ellipse_el, lgrad, dz_sphere = anim.draw_sphere(
            m, p, qo, r_sphere, 
            canvas_center, scale, 
            methane.stroke_color, methane.stroke_thickness, so, 
            svgt.LinearGradientSettings(
                -np.pi/3, 1.0, '#dddddd', '#666666', 'lg', 1.0
            )
        )        
        els.append((dz_sphere,ellipse_el))
        svgels = [
            *zip(
                *sorted(
                    els, key=lambda pair: pair[0]
                )
            )
        ][1]
        doc = svgt.get_svg(width,height)
        svgt.fill_svg(doc,svgels,[lgrad])
        svgt.write_svg(doc,util.fmtf(folder,frame,'svg'))
    print('converting svgs to png')
    util.convert_svgs(folder)
    print('writing gif')
    util.create_gif(folder,gifname='phi-{}-{:03d}-theta-{}-{:03d}'.format(
        'm' if pphi < 0 else 'p', 
        round(np.rad2deg(abs(pphi))),
        'm' if ptheta < 0 else 'p', 
        round(np.rad2deg(abs(ptheta))))
    )
    print('cleaning folder')
    for ext in ('svg','png'):
        util.clean_folder(folder,ext)