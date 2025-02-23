import subprocess as sp, os
from matplotlib import pyplot as plt
from PIL import Image

def run_command(command,folder):
    sp.run(command,shell=True,cwd=folder)

def clean_folder(folder,*exts,prepend=''):
    for ext in exts:
        run_command(['del','{}*.{}'.format(prepend,ext)],folder)

def convert_svgs(folder,prepend='',flip=True,transparent=False,dpi=None):
    run_command(
        [
            'magick',
            'mogrify',
            *(['-flip'] if flip else []),
            *(
                [
                    '-density',
                    '{:d}'.format(dpi),
                    '-units',
                    'PixelsPerInch'
                ] if dpi is not None else []
            ),
            '-format',
            'png',
            *(
                [
                    '-background',
                    'transparent'
                ] if transparent else []
            ),
            '{}*.svg'.format(prepend)
        ],folder
    )

def create_gif(folder,gifname='ani',prepend='',fps=15):
    run_command(
        [
            'magick',
            '-delay',
            '1x{:d}'.format(fps),
            '-dispose',
            '2',
            '{}*.png'.format(prepend), 
            '{}.gif'.format(gifname),
        ],folder
    )

def fmtf(folder,frame,ext,prepend=''):
    return os.path.join(folder,'{}image{:03d}.{}'.format(prepend,frame,ext))

def show_image(fname):
    fig, ax = plt.subplots()
    ax.imshow(Image.open(fname))
    fig.show()