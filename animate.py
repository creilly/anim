import perspective as perp, svgtools as svgt, numpy as np

def draw_sphere(
        m,p,q,r,
        center,scale,
        stroke_color, stroke_thickness, stroke_opacity, 
        lgsettings, 
    ):
    (xbh,xbo,dxb), (ybh,ybo,dyb) = perp.get_ellipse(m,p,q,r)
    xbo, dxb, ybo, dyb = [
        scale * z for z in (xbo, dxb, ybo, dyb)
    ]
    centerp = center + xbo * xbh + ybo * ybh
    ax1, ax2 = dxb * xbh, dyb * ybh
    eg, lg = svgt.get_ellipse(
        centerp, ax1, ax2, np.random.uniform(0,2*np.pi), 
        stroke_color, stroke_thickness, stroke_opacity, 
        lgsettings
    )
    return eg, lg, m.dot(q-p)[2]

def plot_spheres(doc,spheres):
    egs, lgs, dzs = sort_spheres(spheres)
    svgt.fill_svg(doc,egs,lgs)

def sort_spheres(spheres):
    return zip(
        *sorted(
            spheres,key=lambda sphere: sphere[2]
        )
    )