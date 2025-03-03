import perspective as perp, svgtools as svgt, numpy as np, animate as anim, util, jiggle
from matplotlib import pyplot as plt

lc = 1.0 # lattice constant (scene units)
buff = 0.8 # ratio of sphere atom diameter to lattice constant
natoms = 4 # per row

# stroke color, stroke thickness (image units)
stroke_color, stroke_thickness, stroke_opacity = '#444444', 7, 1.0

gstart, gstop = '#ffffff', '#ffd700' # fill colors

# angle of fill gradient, ratio of stop radii to sphere radii
gangle, grad = np.pi / 3, 0.9 

# controls amplitude of jiggles (gives good results)
jiggle_gain = 0.4

def plot_surface(
    canvas_center, # image coords
    surface_center, # scene coords
    m, p, scale, 
    jiggles, 
    stroke_color=stroke_color, stroke_thickness=stroke_thickness, 
    gangle=gangle, grad=grad, gstart=gstart, gstop=gstop    
):
    spheres = []
    for nrow, row in enumerate(qos):
        for ncol, qo in enumerate(row):
            jiggle = jiggles[nrow][ncol]            
            q = qo + np.array(
                [ 0, jiggle, 0 ]
            ) + surface_center
            spheres.append(
                anim.draw_sphere(
                    m,p,q,gold_rad,
                    canvas_center,scale,
                    stroke_color, stroke_thickness, stroke_opacity, 
                    svgt.LinearGradientSettings(
                        gangle, grad, gstart, gstop, 'g{:d}{:d}'.format(nrow,ncol)
                    )
                )
            )   
    return spheres

def generate_gold_jiggles(n,gain=jiggle_gain):
    return jiggle.generate_jiggles(n,(natoms,natoms),gain)

# square grid running parallel to x and z axes, 
# centered at origin ( [0,0,0] )
qos = [
    [
        lc * np.array(
            (col-(natoms-1)/2, 0, row-(natoms-1)/2)
        ) for col in range(natoms)        
    ] for row in range(natoms)
]

gold_rad = lc * buff / 2

if __name__ == '__main__':
    folder = 'gold-images'
    w, h = (1280, 720)
    cx, cy = w/2, h/2
    n = 20 # number of frames # oscillation period about 40
    gain = 0.4 # controls amplitude of vibrations
    scale = 1200 # controls image size

    xlim, ylim, zlim = (
        (-0.5,-4.0), (3,6), (-4,-4)
    )

    ps = np.vstack(
        [
            1/2 * (
                (vmin + vmax) + (vmin - vmax) * np.cos(
                    2 * np.pi * np.arange(n) / n
                )
            )
            for vmin, vmax in (
                xlim, ylim, zlim
            )
        ]
    ).transpose()

    image_center = np.array((cx,cy))
    surface_center = np.array(
        (
            0.0,
            0.0,
            lc * (natoms-1)/2
        )
    )
    print('removing existing image files')
    for ext in ('svg','png'):
        util.clean_folder(folder,ext)
    print('generating jiggles')
    jiggles_series = jiggle.generate_jiggles(2*n,(natoms,natoms),gain)[n:]
    print('generating svg frames')
    for frame in range(n):
        doc = svgt.get_svg(w,h)    
        p = ps[frame]  
        cam_angles = perp.point_camera(p,surface_center)
        m = perp.get_pixel_matrix(*cam_angles)
        spheres = plot_surface(
            image_center,surface_center,
            m, p, scale, 
            jiggles_series[frame], 
            stroke_color, stroke_thickness, 
            gangle, grad, gstart, gstop
        )         
        anim.plot_spheres(doc,spheres)
        svgt.write_svg(doc,util.fmtf(folder,frame,'svg')) 
    print('converting to png')
    util.convert_svgs(folder)
    print('creating gif')
    util.create_gif(folder)
    print('cleaning folder')
    for ext in ('svg',):
        util.clean_folder(folder,ext)