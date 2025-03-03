from xml.etree import ElementTree as et
import numpy as np, util
from matplotlib import pyplot as plt
import colorsys

debug = True
render_stroke = True

# add svg elements `els` to an svg doc, 
# and add any definition elements `defels` 
# to a defs element `defsel`
#
# if defsel not specified will create one and return it
# els should be ordered so that element appearing as top layer is zeroth element
def fill_svg(doc,els,defels,defsel=None):
    if defsel is None:
        defsel = et.Element('defs')
    for defel in defels:
        defsel.append(defel)    
    if defsel not in doc:
        doc.append(defsel)
    for el in els[::-1]:
        doc.append(el)
    return defsel

# generates noise waveform of length n
# that looks pleasing for curves of length n=400
# curve will not greatly exceed er/2=0.75/2
# so will be nice for drawing curves of unit thickness
def generate_waveform(n):
    a = 500
    b = 10
    c = 5

    dr = 0.125
    er = 0.75

    m = 60

    xo = 0
    vo = 0
    xs = []
    vs = []

    while len(xs) < n:        
        f = -a*xo-b*vo+c*(
            (
                (
                    [1,-1][np.random.randint(2)]
                ) * ( 
                    m * np.random.uniform(0.5,1) if np.random.random() < 1/m else 0 
                ) 
            ) if len(xs) else np.random.randint(-m,m+1)
        )
        x = xo + vo/n
        v = vo + f/n
        if abs(x) > er * dr / 2:
            v /= 2
        xs.append(x)
        vs.append(v)
        xo = x
        vo = v
    return np.array(xs) / (dr / 2)

def format_style(styled):
    return ';'.join(
        ':'.join(
            (key,value)
        ) for key, value in styled.items()
    )

def fmt_vals(*l,sep=', ',prec=2):
    return sep.join(
        str(
            v if type(v) is int else
            round(v,prec)
        ) for v in l
    )

def get_gendpoints(center,rad,grad,gangle):      
    dg = np.array(
        [
            rad * grad * fz(gangle)
            for fz in (np.cos,np.sin)
        ]
    )    
    gbeg = center + dg
    gend = center - dg
    return gbeg, gend

# angle : orientation of LG (radians)
# rad   : how far away stops are from center of object (1.0 for stop appearing at edge of circle)
# start : first color of LG
# stop  : second color of LG
# gid   : LG id (for linking to elements)
class LinearGradientSettings:
    def __init__(self,angle,rad,start,stop,gid,opacity=1.0):
        self.angle = angle
        self.rad = rad
        self.start = start
        self.stop = stop
        self.gid = gid
        self.opacity = opacity

# for ellipse of center `center` and mean radius `rad`, 
# create an LG with settings corresponding to `lgsettings`
def generate_lg(center,rad,lgsettings : LinearGradientSettings):
    lgs = lgsettings
    gbeg, gend = get_gendpoints(center,rad,lgs.rad,lgs.angle)
    lg = et.Element(
        'linearGradient',
        attrib={
            **{
                '{}{}'.format(
                    {0:'x',1:'y'}[zindex],{0:'1',1:'2'}[nindex]
                ):fmt_vals(vzn)
                for nindex, vec in enumerate((gbeg,gend)) 
                for zindex, vzn in enumerate(vec)
            },'id':lgs.gid,
            'gradientUnits':'userSpaceOnUse'
        }
    )
    for gcol, goff in ((lgs.start,'0'),(lgs.stop,'1')):
        lg.append(
            et.Element(
                'stop',
                attrib={
                    'style':format_style(
                        {
                            'stop-color':gcol,
                            'stop-opacity':str(round(lgs.opacity,3))
                        }
                    ),
                    'offset':goff
                }
            )
        )
    return lg

# get vector normal to 2d vector `p`
def get_normal(p):
    q = np.cross(
        [0,0,1],[*p,0]
    )[:2]
    return q / np.linalg.norm(q)

# create an ellipse and associated linear gradient fill
# `center` : center of ellipse
# `ax1`    : major axis vector
# `ax2`    : minor axis vector
# to       : starting phase of stroke (radians)
def get_ellipse(
    center,ax1,ax2,to,
    stroke_color,stroke_thickness,stroke_opacity,
    lgsettings : LinearGradientSettings
):
    lgs = lgsettings
    r1, r2 = [
        np.linalg.norm(ax) 
        for ax in (ax1,ax2)
    ]    
    r = np.average((r1,r2))
    angle = np.arctan2(*ax1[::-1])
    lgsp = LinearGradientSettings(
        lgs.angle - angle, lgs.rad, lgs.start, lgs.stop, lgs.gid, lgs.opacity
    )
    lg = generate_lg(center,r,lgsp)    
    ell = et.Element(
        'ellipse',attrib={
            'style':format_style(
                {
                    'stroke':'none',
                    'fill':'url(#{})'.format(lgsettings.gid)                    
                }
            ),**{
                'c{}'.format({0:'x',1:'y'}[zindex]):fmt_vals(cz)
                for zindex, cz in enumerate(center)
            },**{
                'r{}'.format({0:'x',1:'y'}[nindex]):fmt_vals(rn)
                for nindex, rn in enumerate((r1,r2))
            },'transform':'rotate({})'.format(fmt_vals(np.rad2deg(angle),*center,sep=' '))
        }
    )
    group = et.Element('g')
    group.append(ell)    
    n = 400    
    ts = 2 * np.pi * np.arange(n) / n * np.random.uniform(0.85,1.05) + to    
    vs = np.outer(np.cos(ts),ax1) + np.outer(np.sin(ts),ax2) + np.outer(np.ones(len(ts)),np.array(center))
    vps = [
        get_normal(v) for v in [
            np.cos(t)*ax2-np.sin(t)*ax1 
            for t in ts
        ] 
    ]
    if render_stroke:
        path = generate_wavy(vs,stroke_thickness,stroke_color,stroke_opacity,vps)
        group.append(path)
    return group, lg

# helper function for intersection detection
def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# helper function for intersection detection
# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

# generate a "hand-drawn" looking path tracing N x 2 trajectory `vs` (i.e. vs = [(v1x,v1y),(v2x,v2y),...,(vNx,vNy)])
# `vps` is an array of same size as vs and corresponds to an array of N vectors where the nth vector is normal to the tangent of the nth point of the trajectory
# `buffer` scales the random stroke thickness (1.0 is a good starting point)
# `search_length` defines how far the algorithm looks to see if there are any self-intersections of the generated wavy paths
#   when there are tight inside turns of the path, self-intersections can occur
def generate_wavy(vs,st,sc,so=1.0,vps=None,buffer=1.0,search_length=0):
    drs = generate_waveform(len(vs)) * buffer
    if vps is None:
        # approximate using finite diff
        dvs = np.diff(vs,axis=0)
        dvs = 1/2 * (
            np.vstack((dvs,[dvs[-1]]))
            +
            np.vstack(([dvs[0]],dvs))
        )
        vps = [get_normal(dv) for dv in dvs]        
    pps,pms = [
        [
            v + sign * vp * st * (1 + dr) / 2
            for v, vp, dr in zip(vs,vps,drs)
        ] for sign in (+1,-1)
    ]    
    # trim intersections
    if search_length > 0:        
        for arr in (pps,pms):
            i = 0
            j = 2
            while True:                
                if i == (len(arr)-1):
                    break
                if j > (len(arr)-2) or j - i > search_length:                    
                    i += 1
                    j = i + 2
                    continue                       
                if intersect(arr[i], arr[i+1], arr[j], arr[j+1]):                    
                    # remove intersection
                    del arr[i+1:j+1]
                    j = i + 2                    
                else:
                    j += 1
    if debug:
        plt.plot(*zip(*pps),'.')
        plt.plot(*zip(*pms),'o')
        plt.plot(*zip(*vs))        
        if search_length == 0:
            for ps in (pps, pms):
                for v, p in zip(vs,ps):
                    plt.plot(
                        *zip(v,p),color='black',marker='o',ms='3',mfc='none',mec='red'
                    )
        plt.gca().set_aspect('equal')
        plt.show()        
    def fmt_node(code,*coords):
        return '{}{}'.format(
            code,' '.join(
                fmt_vals(*coord,sep=',') for coord in coords
            )
        )
    node_start = fmt_node('M',pps[0])    
    pps = pps[:1 + ((len(pps)-1)//3)*3]
    pms = pms[:1 + ((len(pms)-1)//3)*3][::-1]
    nodes_top = fmt_node('C',*pps[1:])
    rad_fudge = 1.05
    rmid = np.linalg.norm(pms[0] - pps[-1])/2*rad_fudge
    close_index = 0
    node_mid = fmt_node(
        'A',(rmid,rmid),(0,),(0,close_index),pms[0]
    )
    nodes_bot = fmt_node('C',*pms[1:])
    rend = np.linalg.norm(pps[0] - pms[-1])/2*rad_fudge
    node_end = fmt_node(
        'A',(rend,rend),(0,),(0,close_index),pps[0]
    )
    node_close = fmt_node(
        'Z'
    )
    d = ' '.join([node_start,nodes_top,node_mid,nodes_bot,node_end,node_close])
    path = et.Element(
        'path',d=d,
        attrib={
            'fill':sc,'stroke':'None',
            'fill-opacity':str(round(so,3))
        }
    )        
    return path

def get_circle(
    center,r,to,
    stroke_color,stroke_thickness,
    gangle,grad,gstart,gstop,gid
):
    ax1 = r * np.array([1,0])
    ax2 = r * np.array([0,1])
    return get_ellipse(
        center,ax1,ax2,to,
        stroke_color,stroke_thickness,
        gangle,grad,gstart,gstop,gid
    )

def get_svg(w,h):
    return et.Element(
        'svg',
        width=str(w), 
        height=str(h), 
        version='1.1', 
        xmlns='http://www.w3.org/2000/svg'
    )

def get_defs():
    return et.Element('defs')

def write_svg(doc,fname):
    with open(fname,'w') as f:
        et.indent(doc,space=' ')
        f.write(et.tostring(doc).decode())

hmax, smax, lmax, rgbmax = 360, 100, 100, 255
def hsl_to_hex(h,s,l):
    return '#{}'.format(
        '{:02x}' * 3
    ).format(
        *[
            int(round(rgbmax * w))
            for w in colorsys.hls_to_rgb(            
                *[
                    v/vmax for v, vmax in (
                        (h,hmax), (l,lmax), (s,smax)
                    )
                ]
            )
        ]
    )

if __name__ == '__main__':
    w = 480
    h = 360
    doc = get_svg(w,h)
    cx = w/2
    cy = h/2
    r1 = 100
    r2 = 50
    st = 10
    sc = '#cc5555'
    gstart = '#5555cc'
    gstop = '#bbbbbb'
    gangle = np.pi / 3
    gid = 'lg'
    grad = 0.75
    angle = np.pi / 6
    to = np.pi / 4
    ax1 = r1 * np.array([np.cos(angle),np.sin(angle)])
    ax2 = r2 * get_normal(ax1)
    # to see self-intersections:
    #  > set `r2` to 10 and
    #  > set debug to True
    # when you see the plot, zoom in on a tight corner
    ell, lg = get_ellipse(
        (cx,cy),ax1,ax2,to,sc,st,1.0,LinearGradientSettings(gangle,grad,gstart,gstop,gid)
    )
    scale = 100    
    defs = et.Element('defs')
    defs.append(lg)
    doc.append(defs)
    doc.append(ell)
    write_svg(doc,'ell.svg')