import methane, animate, perspective, svgtools, util, pulse
import numpy as np, pprint
from PIL import Image
from matplotlib import pyplot as plt

metascene = 2
metascenesuffix = ''

folder = 'scene2'

single_file = False
scene_giffing = True
metascene_giffing = True

pp = pprint.PrettyPrinter().pprint

width, height = 1280, 720
canvas_shift = 0.1
dpi = 96
fpso = 12
downsample = 1

scale = 1000

MA, MB = 0, 1

mids = (MA, MB)

# methane separation
delta_methane = 2.0
rot_periodo = 5 # seconds
rot_mode = methane.TUMBLE
fillsatmin = 0
fillsatmax = 50
hued = {MA:348, MB:197}
hueo = 142
rchp = 0.5

# camera config
camera_depth = 2.0
camera_height = 2.0
camera_offset = 1.0
camera_drop = 1.0
camera_retreat = -1.5
camera_shift = 1.5
target_advance = -0.25

# pulse config
beg_depth = 10.0
end_depth = -5.0
ampmax = 0.475
ampmin = 0.1
vphi = -np.pi/6
deltav = 0.4
PA, PB = 0, 1
pulse_hues = {PA:265,PB:31} # 63}
pulse_sat, pulse_light = 64, 61
pulse_alpha = 12
psmin, psmax = 3, 16
post_period = 2.0 # seconds
deltalmin = -8.0
delta_ab = 0.1 # transverse displacement of A and B pulses
sigmadeltal = 0.5
to_period = 0.3 # seconds

fps = fpso / downsample
rot_period = rot_periodo * fps
d_rot_angle = 2 * np.pi / rot_period

INTRO, PANOUT, INPHASE, PULSEIN, OUTPHASE, PULSEOUT, INCOH, PULSEINC = (
    ((index+1),name) for index, name in enumerate(
        (
            'intro', 'panout', 
            'in-phase', 'pulse in-phase', 
            'out-of-phase', 'pulse out-of-phase',
            'incoherent', 'pulse incoherent'            
        )
    )
)

scenes = (
    INTRO, PANOUT,
    INPHASE, PULSEIN,
    OUTPHASE, PULSEOUT,
    INCOH, PULSEINC
)

RENDER_ALL = None
scenestorender = RENDER_ALL # (INPHASE,PULSEIN,OUTPHASE,PULSEOUT,INCOH)

ABSORBED, TRANSMITTED, HALF_ABSORBED = 0, 1, 2
pulse_config = {
    PA:[
        (10, ABSORBED), 
        (15.5, TRANSMITTED), 
        (21.0, HALF_ABSORBED)
    ], PB:[
        (11.5, TRANSMITTED), 
        (16.5, ABSORBED), 
        (22.0, HALF_ABSORBED)
    ]    
}

framed = {
    INTRO:3.0, PANOUT:3.5,
    INPHASE:3.7, PULSEIN:2.5,
    OUTPHASE:3.0, PULSEOUT:2.4, 
    INCOH:3.0, PULSEINC:5.0
} # seconds

sceneindexd = sid = {
    scene:scenes.index(scene)
    for scene in scenes
}

framestarts = {
    scene:sum(
        framed[scenes[index]] 
        for index in range(sid[scene])
    ) for scene in scenes
} # seconds

# recompute times to compensate for cut scenes
if scenestorender is not RENDER_ALL:
    for pid, pl in pulse_config.items():
        for index, (pt, pa) in enumerate(pl):
            for scene in scenes:
                tstart = framestarts[scene]
                if tstart > pt:
                    sceneo = scenes[sid[scene]-1]
                    for scenep in scenes:
                        if scenep == sceneo:
                            break
                        if scenep not in scenestorender:                            
                            pt -= framed[scenep]                            
                    break
            pl[index] = (pt,pa)

for scene, scenelength in framed.items(): 
    framed[scene] = int(round(scenelength*fps))

frames = sum(
    framecount for scene, framecount in framed.items() 
    if scenestorender is RENDER_ALL or scene in scenestorender
) # frames

pstart = np.array((camera_offset,camera_height,-camera_depth))
pstop = pstart + np.array((camera_shift,-camera_drop,-camera_retreat))
qstart = np.array((-delta_methane/2,0,0))
qstop = qstart + np.array((0,0,target_advance))

dab = delta_methane * np.array((-1,0,0))
mac = np.zeros(3)
mbc = mac + dab
mcs = {MA:mac, MB:mbc}

def interp_vector(vstart,vstop,x):
    return vstart + (vstop - vstart) * x

def _get_r(scene,sceneframe,vstart,vstop):
    si = sid[scene]
    poi = sid[PANOUT]
    if si < poi:
        return vstart
    if si > poi:
        return vstop
    return vstart + (vstop-vstart) * sceneframe / framed[PANOUT]

def get_p(scene,sceneframe):
    return _get_r(scene,sceneframe,pstart,pstop)

def get_q(scene,sceneframe):
    return _get_r(scene,sceneframe,qstart,qstop)

def get_camera(scene,sceneframe):
    p = get_p(scene,sceneframe)
    q = get_q(scene,sceneframe)
    m = perspective.get_pixel_matrix(*perspective.point_camera(p,q))
    return m, p

huesat_trans = 0.75 # seconds
huesatframemax = hsfm = huesat_trans * fps
def _interp_sat(ss,es,sf,sfo=0,hsfm=hsfm):
    return ss + (es-ss)*(sf-sfo)/hsfm
def get_sat_hue(scene,sceneframe,mid):    
    if scene in (INTRO, PANOUT, PULSEINC): return (
        fillsatmin,hued[mid] if scene == PULSEINC else hueo
    )    
    if scene in (PULSEIN, PULSEOUT): return (
        fillsatmax,{
            PULSEIN:hueo,
            PULSEOUT:hued[mid]
        }[scene]
    )
    for refscene, startsat, endsat in (
        (INPHASE,fillsatmin,fillsatmax),(INCOH,fillsatmax,fillsatmin)
    ):
        if scene == refscene:
            return (
                endsat 
                if sceneframe > hsfm else 
                _interp_sat(startsat,endsat,sceneframe),{
                    INPHASE:hueo,
                    INCOH:hued[mid]
                }[scene]
            )
    if scene == OUTPHASE:
        return (
            fillsatmax 
            if sceneframe > 2*hsfm else (
                _interp_sat(fillsatmax,fillsatmin,sceneframe)
                if sceneframe < hsfm else 
                _interp_sat(fillsatmin,fillsatmax,sceneframe,hsfm)
            ),hueo if sceneframe < hsfm else hued[mid]
        )
def get_hue_sat(scene,sceneframe,mid):
    sat, hue = get_sat_hue(scene,sceneframe,mid)
    return (hue,sat)

rotanglehist = {mid:0.0 for mid in mids}
def _get_rot_angle(scene,sceneframe,mid):
    rot_angle = rotanglehist[mid]
    rotanglehist[mid] += d_rot_angle
    return rot_angle

rotsensed = {MA:+1,MB:-1}
def get_rots(scene,sceneframe,mid):
    rot_angle = _get_rot_angle(scene,sceneframe,mid)
    return [(rotsensed[mid]*methane.axisd[rot_mode],rot_angle)]

jigglesd = {
    mid:methane.generate_methane_jiggles(2*frames, methane.default_jiggle_gain)[-frames:]
    for mid in mids
}

def get_limit(m,p,qo,qhat,zp,axis):
    do = m.dot(qo-p)
    ddq = m.dot(qhat)
    dzp = zp - canvas_center[axis]
    return -(
        dzp * do[2]  - scale * do[axis]
    ) / (
        dzp * ddq[2] - scale * ddq[axis]
    )

vhat = -np.array((np.sin(vphi),0,np.cos(vphi)))
v = deltav * vhat
what = np.array((0,1,0))
vperphat = np.cross(vhat,what)

canvas_center = np.array([d*(1/2+s) for d, s in ((width,0),(height,canvas_shift))])
mstop = perspective.get_pixel_matrix(
    *perspective.point_camera(
        pstop,qstop
    )
)
qo = 1/2*(mac + mbc)
deltalo = get_limit(mstop,pstop,qo,vhat,0,0) + 1.1*deltav*pulse.sigmas*pulse.deltat
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
        m, p = get_camera(scene,sceneframe)
        spheres = {}
        for mid in mids:            
            spheres[mid] = methane.plot_methane(
                canvas_center, 
                mcs[mid], 
                m, p, scale, 
                get_rots(scene, sceneframe, mid), 
                jiggles = jigglesd[mid][frame],                    
                gidp = {MA:'a',MB:'b'}[mid],
                huesat = get_hue_sat(scene, sceneframe, mid),
                rchp = rchp
            )
        pulses = get_pulses(m, p, frame)
        doc = svgtools.get_svg(width,height)
        defs = svgtools.get_defs()
        bels, blgs, bdzs = animate.sort_spheres(spheres[MB])
        svgtools.fill_svg(doc,bels,blgs,defs)
        svgtools.fill_svg(doc,pulses,[],defs)
        aels, algs, adzs = animate.sort_spheres(spheres[MA])
        svgtools.fill_svg(doc,aels,algs,defs)        
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