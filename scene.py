import numpy as np, animate as anim, methane, \
    gold, util, perspective as pers, svgtools as svgt, pulse
from matplotlib import pyplot as plt

G, M, P, A = 0, 1, 2, 3
GIF, SVG, PNG = 0, 1, 2

togen = A
ftype = PNG
folder = 'scene'
n = 150 # 100

w, h = (1280, 720)

scene_shift = -0.25 # vertical

# methane rotation config
nrots = 4
mode = methane.TUMBLE

# camera config (will point towards origin)
px, py, pz = -4, 2, -8
scale = 2000

# methane trajectory config
deltal = 4
deltah = 2
dh = 0.5
ho = 1.1

# surface config
surface_center = np.zeros(3)

# pulse config
vpulse = np.array((-0.25,0,+0.75))
vpulse_hat = vpulse / np.linalg.norm(vpulse)
hpulse = 1
wpulse = np.array((0,1,0))
amppulse = 0.5
dv_start = -5
dv_end = 8
oscillations = 6
pulse_sc = '#842dcc'
pulse_st = 12
pulse_so = 1.0

pco = surface_center + np.array([0,1,0])*hpulse
pcstart = pco + dv_start * vpulse_hat
pcend = pco + dv_end * vpulse_hat

pcs, _ = pulse.get_pulse_coords(pcstart,pcend,n)
deltatw = pulse.deltat / pulse.wiggles
tos = -np.arange(n)/n * oscillations * deltatw

actord = {
    G:'g', M:'m', P:'p', A:'a'
}

alphas = nrots * 2 * np.pi * np.arange(n) / n

canvas_center = np.array((w/2, h * ( 1/2 + scene_shift )))

p = np.array([px,py,pz])

qo = np.zeros(3)

dldn = 2 * deltal / n
dl = 2 * deltal / (deltah/dh + 1)
a = dh / dl**2
dhdlo = 2 * a * dl

ls = -deltal + dldn * np.arange(n)

hs = [
    ho + (
        a * l**2 if abs(l) < dl else
        abs( 
            a * dl**2 + 2 * a * dl * (abs(l) - dl)
        )
    ) for l in ls    
]

qzs = np.zeros(n)
qxs = ls
qys = np.array(hs)

qs = np.vstack(
    (qxs,qys,qzs)
).transpose()

jiggles_sequence = gold.generate_gold_jiggles(2*n,gold.jiggle_gain)[n:]

m = pers.get_pixel_matrix(*pers.point_camera(p,qo))
print('precleaning folder')
prefix = actord[togen]
util.clean_folder(folder,'svg','png',prepend=prefix)
print('generating svgs')
for frame in range(n):   
    spheres = []
    genall = togen == A
    if genall or togen == G: 
        gold_spheres = gold.plot_surface(
            canvas_center, surface_center, 
            m, p, scale, 
            jiggles_sequence[frame]
        )
        spheres.extend(gold_spheres)
    if genall or togen == M:
        methane_spheres = methane.plot_methane(
            canvas_center, qs[frame], 
            m, p, scale, 
            [(methane.axisd[mode],alphas[frame])]
        )
        spheres.extend(methane_spheres)
    if genall or togen == P:
        pulse_path = pulse.plot_pulse(
            canvas_center, scale, 
            pcs[frame], m, p,             
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
    svgt.write_svg(doc,util.fmtf(folder,frame,'svg',prepend=prefix))
if ftype == SVG:
    exit()    
print('converting to png')
util.convert_svgs(folder,transparent=ftype==PNG,prepend=prefix)
if ftype!=SVG:
    print('cleaning svgs')
    util.clean_folder(folder,'svg',prepend=prefix)    
    exit()
print('creating gif')
util.create_gif(folder,prepend=prefix)
print('cleaning pngs')
util.clean_folder(folder,'png',prepend=prefix)