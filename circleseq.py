import svgtools as svgt, numpy as np, subprocess

w = 480
h = 360
cx = w/2
cy = h/2
scale = 100

n = 50

images = []

for m in range(n):        
    doc = svgt.get_svg(w,h)
    circle = svgt.get_circle(
        scale,scale,(cx,cy),np.random.random()*2*np.pi
    )
    doc.append(circle)
    froot = r'newimages\image{:02d}'.format(m)
    fsvg = '{}.svg'.format(froot)
    print('a')
    svgt.write_svg(doc,fsvg)
    print('b')