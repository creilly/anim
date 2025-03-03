import xml.etree.ElementTree as ET, os, svgtools
import util

prefix = 'au'
rootfolder = 'elements'
imagefolder = 'svgs'
folder = os.path.join(rootfolder,prefix)
source = os.path.join(imagefolder,'{}.svg'.format(prefix))
doc = ET.parse(source).getroot()
ns = r'http://www.w3.org/2000/svg'
nsd = {
    'd':ns
}
g = doc.find('d:g',nsd)
rect = g.find('d:rect',nsd)
f = float
prec = 2
def grf(a):
    return round(float(rect.get(a)),prec)
x = grf('x')
y = grf('y')
w = grf('width')
h = grf('height')
r = grf('ry')
t = rect.get('transform')
s = rect.get('style')
g.remove(rect)

sr = str(r)
M, P = '-', ''
lt = (
    (M,1),(M,0),(P,1),(P,0)
)
for n in range(12+1):
    path = ET.Element('ns0:path',nsd)
    xp, yp = x + w - r, y + h
    d = 'm {}'.format(
        ','.join('{:.2f}'.format(v) for v in (xp,yp))
    )
    for m in range(n):
        o = m // 3
        if m % 3 == 0:            
            d += ' a {} 0 0,0 {}'.format(
                ','.join([sr]*2),
                ','.join(
                    '{}{}'.format(sign,sr)
                    for sign in (
                        (M,M), (M,P), (P,P), (P,M)
                    )[o-1]
                )
            )
        else:        
            d += ' l {}'.format(
                ','.join(
                    (
                        '0' if axis != lt[o][1] else '{}{:.2f}'.format(
                            lt[o][0],
                            ((w,h)[axis]-2*r)/2
                        )
                    ) for axis in range(2)                    
                )
            )
    if n == 12:
        d += ' z'
    path.set('d',d)
    path.set('transform',t)
    path.set('style',s)
    path.set('id','path')
    g.append(path)
    svgtools.write_svg(
        doc,util.fmtf(folder,n+1,'svg',prepend=prefix)
    )
    g.remove(path)

util.convert_svgs(folder,flip=False,transparent=True,dpi=400)
util.create_gif(folder)