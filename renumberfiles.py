import os, shutil

folder = r'.\scene1'

scened = {}
for folder, subfolders, files in os.walk(folder):
    for f in files:
        if f.split('.')[-1] == 'png':            
            if f[0] != '_':
                continue
            scene = f[1:][:5]
            if scene not in scened:
                scened[scene] = []
            scened[scene].append(
                (int(f.split('.')[-2][-3:]),f)
            )
for scene, scenel in scened.items():
    for indexp, (indexo,f) in enumerate(sorted(scenel)):
        shutil.copyfile(
            os.path.join(folder,f),
            os.path.join(
                folder,'{}image{:03d}.png'.format(scene,indexp+1)
            )
        )
        os.remove(os.path.join(folder,f))