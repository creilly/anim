import subprocess as sp, os
from matplotlib import pyplot as plt
from PIL import Image

def run_command(command,folder):
    sp.run(command,shell=True,cwd=folder)

# deletes all files in `folder` matching specified file extensions `*exts` 
# and matching file prefix `prepend`
def clean_folder(folder,*exts,prepend=''):
    for ext in exts:
        run_command(['del','{}*.{}'.format(prepend,ext)],folder)

# bulk svg -> png
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

gifprec = 2
def create_gif(folder,gifname='ani',prepend='',fps=12,downsample=None,gifprepend=None):
    if gifprepend is None:
        gifprepend = prepend
    run_command(
        [
            'magick',
            '-delay',
            (
                '1{}x{:d}'.format('0'*gifprec,int(round(10**gifprec*fps)))
                if downsample is None else 
                '{:d}x{:d}'.format(downsample,fps)
            ),
            '-dispose',
            '2',
            '{}*.png'.format(prepend), 
            '{}{}.gif'.format(gifprepend,gifname),
        ],folder
    )

def fmtf(folder,frame,ext,prepend=''):
    return os.path.join(folder,'{}image{:03d}.{}'.format(prepend,frame,ext))

def show_image(fname):
    fig, ax = plt.subplots()
    ax.imshow(Image.open(fname))
    fig.show()