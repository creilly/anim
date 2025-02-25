import perspective as perp, svgtools as svgt, subprocess, animate as anim
import numpy as np, os, util, jiggle, time

rc = 0.3 # scene units 
drhdrc = 0.6/1.1 # scene units
drchdrc = 1.8 # 1.732/1.1 # scene units ( rt(3) = 1.732... )

stroke_color, stroke_thickness = '#444444', 7

cstart, cstop = '#bbbbbb', '#868686'
hstart, hstop = '#ffffff', '#d6d6d6'

gangle, grad = np.pi / 3, 0.9

C, H = 0, 1
nC, nH = 1, 4

species = (C,H)

rh = drhdrc * rc
rch = drchdrc * rc

nSd = {C:nC,H:nH}

# rots: sequence rotations (axis, angle) characterized by rotation axis 
# and rotational angle
# filter: 
lightnessd = {C:53,H:84}
def plot_methane(
    canvas_center, 
    methane_center, 
    m, p, scale, 
    rots, 
    stroke_color=stroke_color, stroke_thickness=stroke_thickness, 
    gangle=gangle, grad=grad, 
    cstart=cstart, cstop=cstop, 
    hstart=hstart, hstop=hstop, 
    jiggles=None,
    salphas=None,
    falphas=None,
    gidp='',
    huesat=None,
    rchp=rch
):
    J, S, F = 0, 1, 2    
    optionsd = {
        atomindex:{
            optionkey:(
                option[atomindex] if option is not None else ([default] * nspecies)
            ) for optionkey, option, default in (
                (J,jiggles,np.zeros(3)),(S,salphas,1.0),(F,falphas,1.0)
            )
        } for atomindex, nspecies in ((C,nC),(H,nH))
    }    
    spheres = []
    if huesat is not None:
        hue, sat = huesat
        cstop, hstop = [
            svgt.hsl_to_hex(hue, sat, lightnessd[s])
            for s in species
        ]
    for atomindex, label, rad, gstart, gstop, vecs in (
        (C,'c', rc, cstart, cstop, cvecs),
        (H,'h', rh, hstart, hstop, hvecs),
    ):
        atomopts = optionsd[atomindex]     
        for vecindex, vec in enumerate(vecs): 
            jiggle_vec = atomopts[J][vecindex]                          
            vec = rchp / rch * np.array(vec) + jiggle_vec
            for (axis, angle) in rots:
                vec = rotate_vec(
                    vec, axis, angle
                )
            q = vec + methane_center            
            spheres.append(
                anim.draw_sphere(
                    m,p,q,rad,
                    canvas_center,scale,
                    stroke_color,stroke_thickness,atomopts[S][vecindex],
                    svgt.LinearGradientSettings(
                        gangle, grad, gstart, gstop, 
                        '{}{}{:d}'.format(gidp,label,vecindex),
                        atomopts[F][vecindex]
                    )
                )
            )
    return spheres

c, s = np.cos, np.sin

cvecs = [
    np.array([0,0,0])
]

hvecs = [
    rch / np.sqrt(3) * np.array(l) for l in [
        [ 1, 1, 1],
        [ 1,-1,-1],
        [-1,-1, 1],
        [-1, 1,-1]
    ]
]

def rotate_vec(vec,axis,angle):    
    return vec + s(angle) * np.cross(axis,vec) + (1 - c(angle)) * np.cross(
        axis, np.cross(axis,vec)
    )

SPIN, TWIRL, TUMBLE = 0, 1, 2
axisd = {
    TWIRL:  np.array([0,1,0])/np.sqrt(1),
    TUMBLE: np.array([1,0,1])/np.sqrt(2),
    SPIN:   np.array([1,1,1])/np.sqrt(3),
}

default_jiggle_gain = 0.1
def generate_methane_jiggles(n,gain=default_jiggle_gain):
    jiggle_tranpose = np.einsum(
        'nmi->mni',jiggle.generate_jiggles(n,(nC + nH,3),gain)
    )    
    cjs = np.einsum(
        'mni->nmi',jiggle_tranpose[:nC]
    )
    hjs = np.einsum(
        'mni->nmi',jiggle_tranpose[nC:]
    )    
    return [
        {
            atomindex:jiggles for atomindex, jiggles in (
                (C,cjs[nn]),(H,hjs[nn])
            )
        } for nn in range(n)
    ]

if __name__ == '__main__':

    folder = 'elims'  

    prefix = 'j'  

    to_clean = ['svg'] # ['svg','png']

    mode = TWIRL

    nphis = 50 # num frames

    jiggle_gain = 0.125

    w, h = (500, 500)
    cx, cy = w/2, h/2
    dpi = 96 # def 96
    
    deltaphi = 0 # 2 * np.pi # total angle of rotation

    alphamin = 1.0
    alphamax = 1.0

    ptheta = 0.0 # np.pi/4 # tilt angle of camera (ptheta > 0 => tilts down)
    pphi = 0.0 # -np.pi/9 # azim angle of camera
    pr = 2.5 # distance of pinhole from mol center (scene units)

    scale = 800

    print('precalculating')
    to = time.time()    

    jiggles = generate_methane_jiggles((3*nphis)//2,jiggle_gain)[-nphis:]
    # print(len(jiggles))
    # import pprint as pp
    # print(pp.PrettyPrinter().pprint(jiggles[0]))
    # exit()

    alphas = [
        {
            ai:[alpha] * nspecies
            for ai, nspecies in ((C,nC),(H,nH))
        } for alpha in 1/2*(alphamin+alphamax) + 1/2*(alphamin-alphamax)*np.cos(
            2 * np.pi * np.arange(nphis)/nphis
        )
    ]

    phis = deltaphi * np.arange(nphis) / nphis

    m = perp.get_pixel_matrix(0,ptheta,pphi)

    # unit vector of where cameras points to
    zphat = m.transpose().dot([0,0,1])

    p = -pr*zphat

    image_center = np.array((cx,cy))
    mol_center = np.zeros(3)

    nhat = axisd[mode]    

    print('precalc time:',round(time.time()-to,1),',','dpi',dpi)    
    to = time.time()
    tg = ts = 0
    for frame, phi in enumerate(phis):
        tgo = time.time()
        doc = svgt.get_svg(w,h)    
        spheres = plot_methane(
            image_center, mol_center, 
            m, p, scale, [
                (nhat,phi)
            ],stroke_color, stroke_thickness, 
            gangle, grad, 
            cstart, cstop, 
            hstart, hstop, falphas=alphas[frame], jiggles=jiggles[frame]
        ) 
        anim.plot_spheres(doc,spheres)
        tgp = time.time()
        tg += tgp - tgo
        tso = time.time()
        svgt.write_svg(doc,util.fmtf(folder,frame,'svg',prefix))
        tsp = time.time()
        ts += tsp - tso
    print('svg gen time:',round(time.time()-to,1),',','dpi',dpi, ',', 'gen time', round(tg,1), 'save time', round(ts,1))
    to = time.time()    
    util.convert_svgs(folder,prefix,dpi=dpi)
    print('svg->png conv time:',round(time.time()-to,1),',','dpi',dpi)
    if nphis > 1:
        to = time.time()    
        util.create_gif(folder,gifname=prefix,prepend=prefix)
        print('png->gif conv time:',round(time.time()-to,1),',','dpi',dpi)
        for ext in to_clean:
            util.clean_folder(folder,ext,prefix)