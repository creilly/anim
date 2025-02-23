import methane, animate, perspective, svgtools, util
import numpy as np, pprint
from PIL import Image

folder = 'scene1'

single_file = True

pp = pprint.PrettyPrinter().pprint

width, height = 1280, 720
dpi = 96
downsample = 8

scale = 800

MA, MB = 0, 1

mids = (MA, MB)

# methane separation
delta_methane = 2.0
rot_period = 45
rot_mode = methane.TWIRL
fillsat = 50
hued = {MA:345, MB:187}

# camera config
camera_depth = 2.0
camera_height = 0.0
camera_retreat = 0.75
camera_rise = 0.75

pstart = np.array((0,0,-camera_depth))
pend = pstart + np.array((-delta_methane/2,camera_rise,-camera_retreat))

qstart = np.zeros(3)
qend = np.array((-delta_methane/2,0,0))

POPIN1, PAUSE1, ROTATE1, PANOUT, POPIN2, ROTATE2 = 1, 2, 3, 4, 5, 6

scenes = (POPIN1, PAUSE1, ROTATE1, PANOUT, POPIN2, ROTATE2)

scenestorender = (ROTATE2,)

framed = {
    POPIN1: 75, PAUSE1: 75, ROTATE1: 75, PANOUT: 45, 
    POPIN2: 30, ROTATE2: 90
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

alphaseq = {
    mid:{
        methane.C:(fc,),
        methane.H:fhs         
    } for mid, (fc, *fhs) in (
        (MA,(0.4, 0.7, 0.8, 0.9, 1.0)),
        (MB,(0.0, 0.25, 0.5, 0.75, 1.0)),
    )
}

alphascened = {MA:POPIN1,MB:POPIN2}
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

rotanglehist = {mid:None for mid in mids}
rotscened = {MA:ROTATE1,MB:ROTATE2}
def _get_rot_angle(scene,frame,mid):
    si = sid[scene]
    rsi = sid[rotscened[mid]]
    if si < rsi:
        return 0.0
    if rotanglehist[mid] == None:
        rotanglehist[mid] = frame
    return 2 * np.pi * (frame - rotanglehist[mid]) / rot_period

rotsensed = {MA:+1,MB:-1}
def get_rots(scene,frame,mid):
    rot_angle = _get_rot_angle(scene,frame,mid)
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
        # check is scene is filtered or reached end of scene
        while scene not in scenestorender or sceneframe == framed[scene]:
            # get next scene index
            sip = sid[scene] + 1
            # if out of scenes, end render
            if sip == len(scenes):
                raise EndRender()            
            scene = scenes[sip]
            sceneframe = 0            
        print(scene,sceneframe)
        m, p = get_camera(scene,sceneframe)
        spheres = []
        for mid in mids:
            alphas = get_alphas(scene,sceneframe,mid)        
            spheres.extend(
                methane.plot_methane(
                    canvas_center, 
                    mcs[mid], 
                    m, p, scale, 
                    get_rots(scene,frame,mid), 
                    jiggles = jigglesd[mid][frame],
                    salphas = alphas, 
                    falphas = alphas, 
                    gidp = {MA:'a',MB:'b'}[mid],
                    huesat = (hued[mid],fillsat)
                )
            )        
        doc = svgtools.get_svg(width,height)
        animate.plot_spheres(doc,spheres)
        svgtools.write_svg(doc,util.fmtf(folder,frame,'svg'))
        if single_file:
            raise EndRender()
        sceneframe += 1
        frame += 1
    except EndRender:
        break
print('converting svg -> png')
util.convert_svgs(folder,dpi=dpi)
if single_file:
    Image.open(util.fmtf(folder,0,'png')).show()
else:
    print('creating gif')
    util.create_gif(folder)