import methane, animate, perspective, svgtools, util, pulse, gold
import numpy as np, pprint
from PIL import Image

np.seterr(all='raise')

metascene = 3
metascenesuffix = ''

folder = 'scene3'

single_file = False
scene_giffing = True
metascene_giffing = True
clean_pngs = True

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
delta_methane = 1.5
rot_periodo = 5 # seconds
rot_mode = methane.TWIRL
fillsatmin = 0
fillsatmax = 50
hued = {MA:348, MB:197}
rchp = 0.5

#trajectory parameters
deltaH = 1.5
deltaL = 2.5
dH = 0.5
Ho = 1.1
tcollision = 11.0
deltatcollision = 2.5
deltatdecohere = 1.25
tcohere = 1.0
deltatcohere = 1.0
decohereseqd = {
    MA:(
        (0.2,273),
        (0.4,175),
        (0.6,93),
        (0.8,347),
        (1.0,104),
    ),MB:(
        (0.25,34),
        (0.50,247),
        (0.75,379),
        (1.00,303),
    )
}

# camera config
p1offset = 2.0
p1above = 1.0
p2above = 1.75
p2tilt = np.pi/4
p3offset = 3.0 # +x
p3above = 0.5
p3shift = 1.5 # -z
deltatpan2 = 5.0
q3shift = 0.25

assert deltatpan2 >= deltatcollision

# pulse config
ampmax = 0.475
ampmin = 0.1
deltav = 0.25
PA, PB = 0, 1
pulse_hues = {PA:265,PB:31}
pulse_sat, pulse_light = 64, 61
pulse_alpha = 1.0
psmin, psmax = 3, 16
post_period = 2.0 # seconds
deltalmin = -8.0
delta_ab = 0.1 # transverse displacement of A and B pulses
sigmadeltal = 0.5
to_period = 0.3 # seconds
vphi = np.pi/4

# surface config
nickel_gstop = '#a5a4c1' # '#3f3e5b' # '#8180a8'
jiggle_suppress = 0.4

fps = fpso / downsample
rot_period = rot_periodo * fps
d_rot_angle = 2 * np.pi / rot_period

INTRO, COHERENT, PANOUT, PAUSE, PAUSE2 = (
    ((index+1),name) for index, name in enumerate(
        (
            'intro', 'coherent', 'panout', 'pause', 'pause2'
        )
    )
)

scenes = (
    INTRO, COHERENT,
    PANOUT, PAUSE, PAUSE2
)

RENDER_ALL = None
scenestorender = RENDER_ALL # (PAUSE,PAUSE2,) # RENDER_ALL

ABSORBED, TRANSMITTED, HALF_ABSORBED = 0, 1, 2
pulse_config = {
    PA:[
        (18.5, HALF_ABSORBED),         
    ], PB:[
        (19.25, HALF_ABSORBED)
    ]    
}

framed = {
    INTRO:1.0, COHERENT:3.0,
    PANOUT:3.0, PAUSE:10.0, PAUSE2:8.0
} # seconds

sceneindexd = sid = {
    scene:scenes.index(scene)
    for scene in scenes
}

frameends = {
    scene:sum(
        framed[scenes[index]] 
        for index in range(sid[scene]+1)
    ) for scene in scenes
} # seconds

# recompute times to compensate for cut scenes
def trim_time(t): 
    if scenestorender is RENDER_ALL:
        return t
    for scene in scenes:
        tend = frameends[scene]        
        if t < tend:            
            for scenep in scenes:
                if scenep == scene:
                    break
                if scenep not in scenestorender:                            
                    t -= framed[scenep]                            
            return t
    return t

if scenestorender is not RENDER_ALL:
    for pid, pl in pulse_config.items():
        for index, (pt, pa) in enumerate(pl):            
            pl[index] = (trim_time(pt),pa)

tcollision = trim_time(tcollision)

for scene, scenelength in framed.items(): 
    framed[scene] = int(round(scenelength*fps))

frames = sum(
    framecount for scene, framecount in framed.items() 
    if scenestorender is RENDER_ALL or scene in scenestorender
) # frames

canvas_center = np.array([d*(1/2+s) for d, s in ((width,0),(height,canvas_shift))])
dab = delta_methane * np.array((0,0,1))
dL = 2 * deltaL / (deltaH/dH + 1)
A = dH / dL**2
def get_methane_center(frame):    
    t = frame/fps    
    dt = max(
        min(
            t - tcollision, +deltatcollision
        ), -deltatcollision
    )    
    L = dt / deltatcollision * deltaL
    H = Ho + (
        A * L**2 if abs(L) < dL else abs(
            A * dL**2 + 2 * A * dL * (abs(L) - dL)
        )
    )
    qo = H * np.array((0,1,0)) + L * np.array((-1,0,0))
    return qo

def get_methane_centers(frame,mid):
    dq = {MA:+1,MB:-1}[mid]*dab/2
    qo = get_methane_center(frame)
    q = qo + dq
    return q

def interp_vector(vstart,vstop,x):
    return vstart + (vstop - vstart) * x
def get_f23(t):
    return (tcollision + t*deltatpan2)*fps
def get_q12(t):
    return q1 + (q2-q1) * (
        6*t**5 - 15*t**4 + 10*t**3
    )
def get_q23(t):
    return interp_vector(q2,get_methane_center(get_f23(t))+dq3,t)
q1 = np.array((deltaL,deltaH+Ho,0))
q2 = np.array((0,Ho,0))
dq3 = np.array((q3shift,0,0))
q3 = get_q23(1)
def get_p12(t):
    return q1 - dab/2 + p1offset*np.array(
        (
            -np.cos(t*(np.pi - p2tilt)),
            0,
            -np.sin(t*(np.pi - p2tilt))
        )
    ) + np.array((0,p1above + (p2above-p1above)*t,0))
def get_p23(t):
    return interp_vector(p2,get_methane_center(get_f23(t))+dp3,t)
p1 = get_p12(0)
p2 = get_p12(1)
dp3 = np.array((+p3offset,+p3above,-p3shift))
p3 = get_p23(1)

def _get_r(scene,sceneframe,frame,v1,v2,v3,interp12,interp23):
    si = sid[scene]
    poi = sid[PANOUT]
    if si < poi:
        return v1
    if si == poi:
        return interp12(sceneframe/framed[PANOUT])
    t = frame / fps
    dt = t - tcollision
    if dt < 0:
        return v2
    if dt > deltatpan2:
        return v3
    return interp23(dt/deltatpan2)

def get_p(scene,sceneframe,frame):    
    return _get_r(scene,sceneframe,frame,p1,p2,p3,get_p12,get_p23)

def get_q(scene,sceneframe,frame):
    return _get_r(scene,sceneframe,frame,q1,q2,q3,get_q12,get_q23)

def get_camera(scene,sceneframe,frame):
    p = get_p(scene,sceneframe,frame)
    q = get_q(scene,sceneframe,frame)
    m = perspective.get_pixel_matrix(*perspective.point_camera(p,q))
    return m, p

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

def get_gold_atoms(m,p,frame):
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
                    gold.gangle, gold.grad, gold.gstart, nickel_gstop, 
                    'g{:d}{:d}'.format(*[n + 1 for n in (nrow,ncol)])
                )                
            )
        )
    gatoms, glgs, dzs = animate.sort_spheres(spheres)
    return gatoms, glgs

def smooth_step(v1,v2,x):
    return v1 + (v2-v1) * (
        3 * x**2 - 2 * x**3
    )

def get_smooth_hue(x,mid):
    huel = decohereseqd[mid]
    hueo = hued[mid]
    xo = 0
    for xp, huep in sorted(huel):
        if x < xp:
            t = (x - xo) / (xp - xo)
            hue = smooth_step(hueo,huep,t)
            hue = hue % 360
            return hue 
        xo = xp
        hueo = huep

def get_smooth_sat(x):
    return smooth_step(fillsatmax,fillsatmin,x)

def get_hue_sat(frame,mid):    
    mhue = hued[mid]
    t = frame/fps
    if t < tcohere:
        return (mhue,fillsatmin)
    dt = t - tcohere
    if dt < deltatcohere:
        return (mhue,fillsatmin + (fillsatmax-fillsatmin)*dt/deltatcohere)    
    dt = t - (tcollision-deltatdecohere)
    if dt < 0:
        return (mhue,fillsatmax)
    if dt < 2*deltatcohere:
        x = dt/(2*deltatcohere)
        hue = get_smooth_hue(x,mid)
        sat = get_smooth_sat(x)
        return (hue,sat)
    return (mhue,fillsatmin)

rotsensed = {MA:+1,MB:-1}
def get_rots(frame,mid):    
    return [(rotsensed[mid]*methane.axisd[rot_mode],d_rot_angle*frame)]

mjigglesd = {
    mid:methane.generate_methane_jiggles(2*frames, methane.default_jiggle_gain)[-frames:]
    for mid in mids
}
gjiggles = gold.generate_gold_jiggles(2*frames,jiggle_suppress*gold.jiggle_gain)[-frames:]

def get_limit(m,p,qo,qhat,zp,axis):
    do = m.dot(qo-p)
    ddq = m.dot(qhat)
    dzp = zp - canvas_center[axis]
    return -(
        dzp * do[2]  - scale * do[axis]
    ) / (
        dzp * ddq[2] - scale * ddq[axis]
    )

vhat = np.array((np.cos(vphi),0,np.sin(vphi)))
v = deltav * vhat
what = np.array((0,1,0))
vperphat = np.cross(vhat,what)

mstop = perspective.get_pixel_matrix(*perspective.point_camera(p3,q3))
qo = get_methane_center(fps*(tcollision+deltatcollision))
deltalo = get_limit(mstop,p3,qo,vhat,width,0) + 1.1*deltav*pulse.sigmas*pulse.deltat

dldt = deltalo / post_period
dtodf = pulse.deltat / pulse.sigmas / to_period / fps
def get_pulses(m,p,frame):
    pulses = []
    for pid, pl in pulse_config.items():
        for crossing_time, absorption in pl:
            current_time = frame / fps
            deltat = current_time - crossing_time
            deltal = dldt * deltat
            ao = ampmax
            ap = {
                ABSORBED:ampmin,
                TRANSMITTED:ampmax,
                HALF_ABSORBED:1/3*(2*ampmin+ampmax)
            }[absorption]
            st = psmax + (psmin - psmax) * (deltal-deltalo) / (
                deltalmin - deltalo
            )
            if deltal > deltalo or deltal < deltalmin:
                continue
            if abs(deltal) > sigmadeltal:
                amp = ap if deltat > 0 else ao
            else:
                amp = ap + (ao - ap) * (
                    sigmadeltal - deltal
                ) / (2 * sigmadeltal)
            q = qo + deltal * vhat + {
                PA:+1,PB:-1
            }[pid] * vperphat * delta_ab
            dz = m.dot(q-p)[2]
            pulses.append(
                (
                    (pid,dz), pulse.plot_pulse(
                        canvas_center, scale, 
                        q, m, p, 
                        v, what,
                        amp, dtodf*frame, 
                        svgtools.hsl_to_hex(pulse_hues[pid],pulse_sat,pulse_light), 
                        st, pulse_alpha
                    )
                )
            )
    if pulses:
        _, sorted_pulses = zip(*sorted(pulses))
        return sorted_pulses
    else: return []

print('cleaning folder')
util.clean_folder(folder,'svg','png')
print('generating svgs')
class EndRender(Exception): pass
def get_prefix(scenenum): 
    return 'S{:d}{}s{:02d}'.format(metascene,metascenesuffix,scenenum)
scene = scenes[0]
sceneframe = frame = 0
while True:
    try:        
        scenenum, scenename = scene
        prefix = get_prefix(scenenum)
        skipping_scene = (
            scenestorender is not RENDER_ALL and 
            scene not in scenestorender 
        )
        single_file_end = single_file and frame == 1
        scene_finished = sceneframe == framed[scene]
        # check is scene is filtered or reached end of scene
        if skipping_scene or single_file_end or scene_finished:
            if sceneframe > 0:
                print('converting scene {} svg -> png'.format(scenename))
                util.convert_svgs(folder,dpi=dpi,prepend=prefix)
                if scene_giffing:
                    print('creating scene {} gif'.format(scenename))
                    util.create_gif(folder,fps=fps,prepend=prefix)
                print('deleting scene {} svgs'.format(scenename))
                util.clean_folder(folder,'svg',prepend=prefix)
            # get next scene index
            sip = sid[scene] + 1
            # if out of scenes, end render
            if sip == len(scenes) or single_file_end:
                raise EndRender()  
            scene = scenes[sip]
            sceneframe = 0
            continue     
        print(str(scenenum).rjust(3), scenename.ljust(20),str(sceneframe).rjust(3))
        m, p = get_camera(scene,sceneframe,frame)
        gatoms, glgs = get_gold_atoms(m,p,frame)
        spheres = {}
        for mid in mids:            
            spheres[mid] = methane.plot_methane(
                canvas_center, 
                get_methane_centers(frame,mid), 
                m, p, scale, 
                get_rots(frame, mid), 
                jiggles = mjigglesd[mid][frame],                    
                gidp = {MA:'a',MB:'b'}[mid],
                huesat = get_hue_sat(frame,mid),
                rchp = rchp
            )
        pulses = get_pulses(m, p, frame)
        doc = svgtools.get_svg(width,height)
        defs = svgtools.get_defs()
        svgtools.fill_svg(doc,gatoms,glgs)
        aels, algs, adzs = animate.sort_spheres(spheres[MA])
        svgtools.fill_svg(doc,aels,algs,defs)        
        svgtools.fill_svg(doc,pulses,[],defs)
        bels, blgs, bdzs = animate.sort_spheres(spheres[MB])
        svgtools.fill_svg(doc,bels,blgs,defs)
        svgtools.write_svg(doc,util.fmtf(folder,sceneframe+1,'svg',prepend=prefix))
        sceneframe += 1
        frame += 1
    except EndRender:
        break
if single_file:
    Image.open(util.fmtf(folder,0,'png',prepend=prefix)).show()
elif metascene_giffing:
    gifprefix = get_prefix(0)
    pngprefix = gifprefix[:2]
    print('creating composite gif')
    util.create_gif(folder,prepend=pngprefix,gifprepend=gifprefix,fps=fps)
if clean_pngs:
    print('cleaning pngs')
    util.clean_folder(folder,'png')