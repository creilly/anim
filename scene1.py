import methane, animate, perspective, svgtools, util
import numpy as np, pprint
from PIL import Image
from matplotlib import pyplot as plt

metascene = 1

folder = 'scene1'

single_file = False

pp = pprint.PrettyPrinter().pprint

width, height = 1280, 720
dpi = 96
fpso = 12
downsample = 1

scale = 800

MA, MB = 0, 1

mids = (MA, MB)

# methane separation
delta_methane = 2.0
rot_periodo = 3 # seconds
rot_mode = methane.TWIRL
fillsat = 50
hued = {MA:348, MB:197}

# camera config
camera_depth = 2.0
camera_height = 0.0
camera_retreat = 0.75
camera_rise = 0.75

fps = fpso / downsample
rot_period = rot_periodo * fps
d_rot_angle = 2 * np.pi / rot_period

pstart = np.array((0,0,-camera_depth))
pend = pstart + np.array((-delta_methane/2,camera_rise,-camera_retreat))

qstart = np.zeros(3)
qend = np.array((-delta_methane/2,0,0))

POPIN1, PAUSE1, ROTATE1, \
    PANOUT, POPIN2, PAUSE2, ROTATE2, \
     COHERENT, FLUCTUATING, INCOHERENT = (
    ((index+1),name) for index, name in enumerate(
        (
            'popin1', 'pause1', 'rotate1', 
            'panout', 'popin2', 'pause2', 'rotate2', 
            'coherent', 'fluctuating', 'incoherent' 
        )
    )
)

scenes = (
    POPIN1, PAUSE1, ROTATE1, 
    PANOUT, POPIN2, PAUSE2, ROTATE2, 
    COHERENT, FLUCTUATING, INCOHERENT
)

RENDER_ALL = None
scenestorender = RENDER_ALL # (ROTATE1, PANOUT,POPIN2,PAUSE2,ROTATE2)

framed = {
    POPIN1: 73, PAUSE1: 85, ROTATE1: 85, PANOUT: 24, 
    POPIN2: 15, PAUSE2: 72, ROTATE2: 120, COHERENT: 250, FLUCTUATING: 70, 
    INCOHERENT: 70
}

for scene, scenelength in framed.items(): 
    framed[scene] = scenelength // downsample

sceneindexd = sid = {
    scene:scenes.index(scene)
    for scene in scenes
}

frames = sum(framed.values())

dab = delta_methane * np.array((-1,0,0))
mac = np.zeros(3)
mbc = mac + dab
mcs = {MA:mac, MB:mbc}

def get_camera(scene,sceneframe):
    si = sid[scene]
    poi = sid[PANOUT]
    if si < poi:
        p, q = pstart, qstart
    if si > poi:
        p, q = pend, qend
    if si == poi:
        p, q = (
            rstart + (rend - rstart) * sceneframe / framed[PANOUT] 
            for rstart, rend in (
                (pstart, pend), (qstart, qend)
            )
        )
    m = perspective.get_pixel_matrix(*perspective.point_camera(p,q))
    return m, p

alphascened = {MA:POPIN1,MB:POPIN2}
alphaseq = {
    mid:{
        methane.C:(fc,),
        methane.H:fhs         
    } for mid, (fc, *fhs) in {
        mid:np.array(secs) * fps / framed[alphascened[mid]]
        for mid, secs in {
            MA:(3.0, 4.8, 5.05, 5.30, 5.55),
            MB:(0.0, 0.25, 0.5, 0.75, 1.0)
        }.items()
    }.items()    
}

def _get_alpha(scene,sceneframe,mid,species,atom):
    si = sid[scene]
    ts = alphascened[mid]
    ti = sid[ts]
    if si < ti:
        return 0.0
    if si > ti:
        return 1.0
    if si == ti:
        sff = sceneframe / framed[ts]
        start_sff = alphaseq[mid][species][atom]        
        return 1.0 if sff > start_sff else 0.0
        
def get_alphas(scene,sceneframe,mid):
    return {
        species:[
            _get_alpha(scene,sceneframe,mid,species,atom)
            for atom in range(
                methane.nSd[species]
            )
        ] for species in methane.species
    }

sat_trans = 30 # frames
sat_max = 50
def get_sat(scene,sceneframe,mid):    
    if scene not in (COHERENT,FLUCTUATING):
        return 0
    if scene == COHERENT:
        return int(
            round(
                sat_max * (
                1 if sceneframe > sat_trans else 
                sceneframe / sat_trans
                )
            )
        )
    if scene == FLUCTUATING:
        sceneframes = framed[FLUCTUATING]
        framestogo = ftg = sceneframes - sceneframe
        return int(
            round(
                sat_max * (
                    1 if ftg > sat_trans else 
                    ftg / sat_trans
                )
            )
        )
hues = {
    MA:(
        (hued[MA],0.00),
        (120,0.20),
        (240,0.40),
        (319,0.60),
        (721,0.80),
        (361,1.00)
    ),MB:(
        (hued[MB],0.00),
        (309,0.33),
        (86,0.67),
        (230,1.00),
    )
}
def _get_subhue(sceneframe,mid):
    f = sceneframe / framed[FLUCTUATING]
    huel = hues[mid]
    hueindex = 1
    while True:
        huep, fp = huel[hueindex]        
        if f < fp:
            hueo, fo = huel[hueindex-1]    
            dfnorm = (f - fo)/(fp-fo)
            hue = hueo + (huep - hueo) * (
                3 * dfnorm**2 - 2 * dfnorm**3
            )
            return int(round(hue % 360))
        hueindex += 1
# for mid in mids:
#     plt.plot(
#         [
#             _get_subhue(sceneframe,mid) 
#             for sceneframe in range(framed[FLUCTUATING])
#         ],label=str(mid)
#     )
# plt.legend()
# plt.show()
# exit()

def get_hue(scene,sceneframe,mid):
    if scene != FLUCTUATING:
        return hued[mid]
    return _get_subhue(sceneframe,mid)

def get_hue_sat(scene,sceneframe,mid):
    hue = get_hue(scene,sceneframe,mid)
    sat = get_sat(scene,sceneframe,mid)    
    return (hue,sat)

rotanglehist = {mid:None for mid in mids}
rotscened = {MA:ROTATE1,MB:ROTATE2}
rot2waitd = {MA:0.0,MB:1.5} # seconds
def _get_rot_angle(scene,sceneframe,mid):
    si = sid[scene]
    rsi = sid[rotscened[mid]]
    psi = sid[PAUSE2]
    r2si = sid[ROTATE2]
    if si < rsi:
        return 0.0
    if rotanglehist[mid] == None:
        rotanglehist[mid] = 0.0
    rotangle = rotanglehist[mid]
    if (
        si == psi 
        or 
        (
            si == r2si and sceneframe < fps*rot2waitd[mid]
        )
    ):
        # MA spins until back to where it started, so it doesn't get locked in bad position
        if mid == MB or (
            min(
                rotangle % (2*np.pi),
                2 * np.pi - (rotangle % (2*np.pi)),
            ) <= d_rot_angle
        ):
            return rotanglehist[mid]
    rotanglehist[mid] += d_rot_angle
    return rotanglehist[mid]

rotsensed = {MA:+1,MB:-1}
def get_rots(scene,sceneframe,mid):
    rot_angle = _get_rot_angle(scene,sceneframe,mid)
    return [(rotsensed[mid]*methane.axisd[rot_mode],rot_angle)]

jigglesd = {
    mid:methane.generate_methane_jiggles(frames) 
    for mid in mids
}

canvas_center = np.array([d/2 for d in (width,height)])

scene = scenes[0]
sceneframe = sceneframe = 0
frame = 0

util.clean_folder(folder,'svg','png')
print('generating svgs')
class EndRender(Exception): pass
while True:
    try:        
        scenenum, scenename = scene
        prefix = 'S{:d}s{:02d}'.format(metascene,scenenum)
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
                print('creating scene {} gif'.format(scenename))
                util.create_gif(folder,fps=15//downsample,prepend=prefix)
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
        spheres = []
        for mid in mids:
            alphas = get_alphas(scene,sceneframe,mid)        
            spheres.extend(
                methane.plot_methane(
                    canvas_center, 
                    mcs[mid], 
                    m, p, scale, 
                    get_rots(scene, sceneframe, mid), 
                    jiggles = jigglesd[mid][frame],
                    salphas = alphas, 
                    falphas = alphas, 
                    gidp = {MA:'a',MB:'b'}[mid],
                    huesat = get_hue_sat(scene, sceneframe, mid)
                )
            )        
        doc = svgtools.get_svg(width,height)
        animate.plot_spheres(doc,spheres)        
        svgtools.write_svg(doc,util.fmtf(folder,frame,'svg',prepend=prefix))
        sceneframe += 1
        frame += 1
    except EndRender:
        break
if single_file:
    Image.open(util.fmtf(folder,0,'png',prepend=prefix)).show()
