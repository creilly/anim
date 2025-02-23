import numpy as np, animate as anim, methane, \
    gold, util, perspective as pers, svgtools as svgt, pulse
from matplotlib import pyplot as plt

G, M, P, A = 0, 1, 2, 3
GIF, SVG, PNG = 0, 1, 2

togen = P
ftype = PNG
folder = 'scene'
n = 50

w, h = (1280, 720)

scene_shift = 0.0 # vertical

# methane rotation config
nrots = 4
mode = methane.TUMBLE

# camera config (will point towards origin)
px, py, pz = 0, 0, -2
scale = 500

# pulse config
vpulse = np.array((1,0,1)) / np.sqrt(2)
vpulse_hat = vpulse / np.linalg.norm(vpulse)
wpulse = np.array((0,1,0))
amppulse = 0.5
dv_start_min = -2
dv_end = 2
oscillations = 6
pulse_sc = '#842dcc'
pulse_st = 12
pulse_so = 1.0
# by what distance the pulse should be entirely in front of the camera to start
pz_buffer = 0.5

###################
### precomputations
###################

# camera location
p = np.array([px,py,pz])

# scene center
qo = np.zeros(3)

# camera pointing at scene center
m = pers.get_pixel_matrix(*pers.point_camera(p,qo))

# pulse crossing point
pco = np.zeros(3)

dv_flush = -(
    m.dot(
        qo - pulse.sigmas * pulse.deltat * vpulse - p
    ).dot([0,0,1]) - ( pz + pz_buffer )
) / m.dot(vpulse).dot([0,0,1])    

dv_start = max(dv_flush, dv_start_min)

pcstart = pco + dv_start * vpulse_hat
pcend = pco + dv_end * vpulse_hat

pqs, _ = pulse.get_pulse_coords(pcstart,pcend,n)
deltatw = pulse.deltat / pulse.wiggles
tos = -np.arange(n)/n * oscillations * deltatw

actord = {
    G:'g', M:'m', P:'p'
}

alphas = nrots * 2 * np.pi * np.arange(n) / n

canvas_center = np.array((w/2, h * ( 1/2 + scene_shift )))

print('precleaning folder')
prefix = actord[togen]
for ext in ('svg','png'):
    util.clean_folder(folder,ext,prefix)
print('generating svgs')
for frame in range(n):   
    spheres = []
    genall = togen == A
    if genall or togen == M:
        methane_spheres = methane.plot_methane(
            canvas_center, qo, 
            m, p, scale, 
            [(methane.axisd[mode],alphas[frame])]
        )
        spheres.extend(methane_spheres)
    if genall or togen == P:
        
        pulse_path = pulse.plot_pulse(
            canvas_center, pcs[frame], 
            m, p, scale, 
            vpulse, wpulse, amppulse, tos[frame], 
            pulse_sc, pulse_st, pulse_so
        )
    else:
        pulse_path = None
    doc = svgt.get_svg(w,h)
    if spheres:
        anim.plot_spheres(doc,spheres)
    if pulse_path is not None:
        doc.append(pulse_path)
    svgt.write_svg(doc,util.fmtf(folder,frame,'svg',prefix))
if ftype == SVG:
    exit()    
print('converting to png')
util.convert_svgs(folder,transparent=ftype==PNG)
if ftype!=SVG:
    print('cleaning svgs')
    util.clean_folder(folder,'svg',prefix)    
    exit()
print('creating gif')
util.create_gif(folder)
print('cleaning pngs')
util.clean_folder(folder,'png',prefix)