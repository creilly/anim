import methane, animate, perspective, svgtools, util, pulse, gold, colorsys
import numpy as np, pprint
from PIL import Image

debug = False

np.seterr(all='raise')

metascene = 5
metascenesuffix = ''

folder = 'scene5'

single_file = False
scene_giffing = True
metascene_giffing = True
clean_imgs = True

pp = pprint.PrettyPrinter().pprint

width, height = 1280, 720
canvas_shift = 0.0
dpi = 96
fpso = 12
downsample = 8

scale = 700

MA, MB = 0, 1

mids = (MA, MB)

# methane parameters
rchp = 0.5
atom_omit = 2 # (-1,-1,+1) coords
stroke_opacity = 1.0

# traj params
ttraj = 8.0 # seconds
tdetach = 3.0
dtdetach = 2.0
vtraj = 0.75
vtrajtheta = np.pi/6
vtrajphi = -np.pi/2
dtrot = 4.0
ho = 5.0
dhp = 0.125
hpo = gold.gold_rad + dhp
hpc = hpo + methane.rc
hph = hpo + methane.rh

# camera config
dpxm = 2.5
dpyp = 1.5
dpzm = 2
dqyp = 1.0

# surface config
jiggle_suppress = 1.0
nrow = 2 # z
ncol = 2 # x
drow = -1 # z
dcol = 0 # x

fps = fpso / downsample

frames = int(round(ttraj *fps))

canvas_center = np.array([d*(1/2+s) for d, s in ((width,canvas_shift),(height,0))])

dq = np.array((0,+dqyp,0))
qo = gold.qos[nrow][ncol]
qop = gold.qos[nrow+drow][ncol+dcol]
q = qo + np.array((0,+hpc,0))
qp = qop + np.array((0,+hph,0))
po = qo
dp = np.array((-dpxm,+dpyp,-dpzm))
p = po + dp
m = perspective.get_pixel_matrix(*perspective.point_camera(p,q))

def filter_gold(m,p):
    indices = []
    for nrow in range(gold.natoms):
        for ncol in range(gold.natoms):
            q = gold.qos[nrow][ncol]
            dx, dy, dz = m.dot(q-p)
            yp = canvas_center[1] + scale * dy / dz            
            if dz > 0:                
                if yp < 2 * height and yp > -height:
                    indices.append((nrow,ncol))
    return indices

def get_gold_atoms(frame):
    spheres = []    
    for nrow, ncol in filter_gold(m,p):        
        qo = gold.qos[nrow][ncol]
        dq = gjiggles[frame][nrow][ncol]
        q = qo + dq * np.array((0,1,0))
        spheres.append(
            animate.draw_sphere(
                m, p, q, gold.gold_rad, 
                canvas_center, scale, gold.stroke_color, 
                gold.stroke_thickness, gold.stroke_opacity, 
                svgtools.LinearGradientSettings(
                    gold.gangle, gold.grad, gold.gstart, gold.gstop, 
                    'g{:d}{:d}'.format(*[n + 1 for n in (nrow,ncol)])
                )                
            )
        )
    gatoms, glgs, dzs = animate.sort_spheres(spheres)
    return gatoms, glgs, dzs

def smooth_step(x):
    return 3 * x**2 - 2 * x**3

raxis = np.array((1,0,1))/np.sqrt(2)
tangle = np.arccos(-1/3)
phi = tangle/2
v = np.array(
    (
        np.sin(vtrajtheta)*np.sin(vtrajphi),
        np.cos(vtrajtheta),
        np.sin(vtrajtheta)*np.cos(vtrajphi),
    )
) / fps # scene units per frame
w = 2 * np.pi / dtrot / fps # radians per frame
dr = np.zeros(3)

mjiggles = methane.generate_methane_jiggles(2*frames, methane.default_jiggle_gain)[-frames:]
gjiggles = gold.generate_gold_jiggles(2*frames,jiggle_suppress*gold.jiggle_gain)[-frames:]

print('cleaning folder')
util.clean_folder(folder,'svg','png')
print('generating svgs')

class EndRender(Exception): pass
prefix = 'S{:d}'.format(metascene)
frame = 0
while True:
    try:        
        # check is metascene is finished
        if (single_file and frame) or frame == frames:            
            raise EndRender()
        print('{: 4d}'.format(frame+1),'/','{: 4d}'.format(1 if single_file else frames))         
        doc = svgtools.get_svg(width,height)
        defs = svgtools.get_defs()        
        t = frame/fps
        dt = t-tdetach
        if t > tdetach:            
            vp = v
            wp = w
            if dt < dtdetach:
                x = dt / dtdetach
                vp = vp * smooth_step(x)
                wp = wp * smooth_step(x)
            dr += vp
            phi += wp
        else: 
            vp = 0
            wp = 0
        els = []
        lgs = []
        for species in methane.species:
            atom_radius = {methane.C:methane.rc,methane.H:methane.rh}[species]
            for atomindex in range(methane.nSd[species]):                
                Qo = q + dr
                jiggle = mjiggles[frame][species][atomindex]
                dQ = {
                    methane.C:methane.cvecs,
                    methane.H:methane.hvecs
                }[species][atomindex]*rchp/methane.rch                
                dQ = dQ + jiggle
                dQ = methane.rotate_vec(dQ,raxis,phi)
                Q = Qo + dQ
                if species == methane.H and atomindex == atom_omit:
                    Qpo = qp
                    dQp = jiggle
                    Qp = Qpo + dQp
                    if t < tdetach:
                        Q = Qp
                    elif dt > dtdetach:
                        Q = Q
                    else:
                        Q = Qp + (Q - Qp)*smooth_step(dt/dtdetach)
                el, lg, dz = animate.draw_sphere(
                    m, p, Q, atom_radius, 
                    canvas_center, scale, 
                    methane.stroke_color, methane.stroke_thickness, stroke_opacity, 
                    svgtools.LinearGradientSettings(
                        methane.gangle, methane.grad, {
                            methane.C:methane.cstart, 
                            methane.H:methane.hstart
                        }[species], {
                            methane.C:methane.cstop, 
                            methane.H:methane.hstop
                        }[species], 'm{}{:d}'.format(
                            {methane.C:'c',methane.H:'h'}[species],atomindex
                        )
                    )                    
                )                   
                els.append((dz,el))
                lgs.append(lg)
        gels, glgs, gdzs = get_gold_atoms(frame)
        lgs.extend(glgs)
        els.extend(zip(gdzs,gels))
        dzs, sorted_els = zip(*sorted(els,key = lambda pair: pair[0]))        
        svgtools.fill_svg(doc,sorted_els,lgs,defs)
        svgtools.write_svg(doc,util.fmtf(folder,frames-frame,'svg',prepend=prefix))        
        frame += 1
    except EndRender:
        break
print('converting svg -> png')
util.convert_svgs(folder,dpi=dpi,prepend=prefix)
print('cleaning svgs')
# util.clean_folder(folder,'svg')
if single_file:
    Image.open(util.fmtf(folder,frames,'png',prepend=prefix)).show()
else:
    print('creating composite gif')
    util.create_gif(folder,prepend=prefix,gifprepend=prefix,fps=fps)
if clean_imgs:
    print('cleaning pngs and svgs')
    util.clean_folder(folder,'png','svg')