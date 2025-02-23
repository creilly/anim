import numpy as np, animate as anim, methane, \
    gold, util, perspective as pers, svgtools as svgt, \
    jiggle
from matplotlib import pyplot as plt

folder = 'scene'

w, h = (1280, 720)

scene_shift = -0.25 # vertical

nrots = 4
mode = methane.TUMBLE

px, py, pz = -4, 2, -8

deltal = 4
deltah = 2
dh = 0.5
ho = 1.1
n = 100
scale = 2000

alphas = nrots * 2 * np.pi * np.arange(n) / n

canvas_center = np.array((w/2, h * ( 1/2 + scene_shift )))
surface_center = np.zeros(3)

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

jiggles_sequence = gold.generate_jiggles(2*n)[n:]

m = pers.get_pixel_matrix(*pers.point_camera(p,qo))
print('precleaning folder')
for ext in ('svg','png'):
    util.clean_folder(folder,ext)
print('generating svgs')
for frame in range(n):    
    gold_spheres = gold.plot_surface(
        canvas_center, surface_center, 
        m, p, scale, jiggles_sequence[frame]
    )
    methane_spheres = methane.plot_methane(
        canvas_center, qs[frame], 
        m, p, scale, 
        [(methane.axisd[mode],alphas[frame])]
    )
    doc = svgt.get_svg(w,h)    
    anim.plot_spheres(
        doc, gold_spheres + methane_spheres
    )
    svgt.write_svg(doc,util.fmtf(folder,frame,'svg'))
print('converting to png')
util.convert_svgs(folder)
print('creating gif')
util.create_gif(folder)
print('cleaning folder')
for ext in ('svg',):
    util.clean_folder(folder,ext)