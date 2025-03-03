import methane, animate, perspective, svgtools, util
import numpy as np, pprint
from PIL import Image
from matplotlib import pyplot as plt

metascene = 1

folder = 'scene1'

single_file = False
clean_pngs = True

pp = pprint.PrettyPrinter().pprint

width, height = 1280, 720
dpi = 96
fpso = 12
downsample = 16

scale = 800

MA, MB = 0, 1

mids = (MA, MB)

# methane separation
delta_methane = 2.0
rot_periodo = 4.0 # seconds
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
scenestorender = RENDER_ALL # (PANOUT,POPIN2,PAUSE2,ROTATE2)

framed = {
    POPIN1: 73, PAUSE1: 71, ROTATE1: 74, PANOUT: 24, 
    POPIN2: 15, PAUSE2: 61, ROTATE2: 114, COHERENT: 252, FLUCTUATING: 60, 
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
            MA:(2.8, 5.0, 5.25, 5.5, 5.75),
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

def smooth_step(v1,v2,x):
    return v1 + (v2-v1) * (
        3 * x**2 - 2 * x**3
    )

sat_trans = 30 # frames
sat_max = 50
sat_over = 80
sat_none = 0
emphd = {MA:93, MB:126}
emph_trans = 12
def get_sat(scene,sceneframe,mid):
    if scene not in (COHERENT,FLUCTUATING):
        return sat_none
    if scene == COHERENT:
        if sceneframe < sat_trans:            
            return sat_max * sceneframe / sat_trans
        empha, emphb = [
            emph - emph_trans for emph in (emphd[MA], emphd[MB])
        ]
        if sceneframe < empha:
            return sat_max
        dsf = sceneframe - empha        
        if dsf > (emphb - empha):
            dsf = sceneframe - emphb
            sat_crit = {
                MB:sat_over,MA:sat_none
            }[mid]
        else:
            sat_crit = {
                MA:sat_over,MB:sat_none
            }[mid]
        if dsf < emph_trans:
            return smooth_step(sat_max,sat_crit,dsf/emph_trans)
        if dsf < 2 * emph_trans:
            return smooth_step(sat_crit,sat_max,(dsf-emph_trans)/emph_trans)
        return sat_max        
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

rotanglehist = {mid:phio for mid, phio in ((MA,0.0),(MB,np.pi*10.5/22))}
rotscened = {MA:ROTATE1,MB:ROTATE2}
rot2waitd = {MA:0.0,MB:0.75} # seconds
def _get_rot_angle(scene,sceneframe,mid):
    rotangle = rotanglehist[mid]
    si = sid[scene]
    rsi = sid[rotscened[mid]]
    if si < rsi:
        return rotangle
    pausescenes = (POPIN2,PAUSE2)
    if scene in pausescenes:
        return rotangle    
    if scene == ROTATE2 and sceneframe < fps*rot2waitd[mid]:
        return rotangle
    rotangle = rotanglehist[mid] = rotangle + d_rot_angle
    return rotangle

rotsensed = {MA:+1,MB:-1}
def get_rots(scene,sceneframe,mid):
    rot_angle = _get_rot_angle(scene,sceneframe,mid)
    return [(rotsensed[mid]*methane.axisd[rot_mode],rot_angle)]

jigglesd = {
    mid:methane.generate_methane_jiggles(frames) 
    for mid in mids
}

canvas_center = np.array([d/2 for d in (width,height)])

util.clean_folder(folder,'svg','png')
print('generating svgs')
class EndRender(Exception): pass
def get_prefix(scenenum): 
    return 'S{:d}s{:02d}'.format(metascene,scenenum)
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
                print('creating scene {} gif'.format(scenename))
                util.create_gif(folder,fps=fpso,downsample=downsample,prepend=prefix)
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
        svgtools.write_svg(doc,util.fmtf(folder,sceneframe+1,'svg',prepend=prefix))
        sceneframe += 1
        frame += 1
    except EndRender:
        break
if single_file:
    Image.open(util.fmtf(folder,1,'png',prepend=prefix)).show()
gifprefix = get_prefix(0)
pngprefix = gifprefix[:2]
print('creating composite gif')
util.create_gif(folder,prepend=pngprefix,gifprepend=gifprefix,fps=fpso,downsample=downsample)
if clean_pngs:
    print('cleaning all pngs')
    util.clean_folder(folder,'png')