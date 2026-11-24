"""Cache the fitted splats on inverse-depth planes for low-cost browser parallax."""
from pathlib import Path
from tempfile import TemporaryDirectory
import argparse
import json
import subprocess
from PIL import Image

def bake(binary, input_file, directory):
    with TemporaryDirectory() as temp:
        subprocess.run([binary,input_file,temp],check=True)
        pieces=[]
        for file in sorted(Path(temp).glob('*.rgba')):
            i=int(file.stem)
            depth=1/(.14+i*(2.9-.14)/31)
            img=Image.frombytes('RGBA',(768,576),file.read_bytes())
            # Split into occupied tiles so hidden areas consume neither texture
            # memory nor fill rate. One pixel of real apron prevents seams.
            for top in range(0,576,144):
                for left in range(0,768,192):
                    tile=img.crop((left,top,left+192,top+144))
                    rect=tile.getchannel('A').getbbox()
                    if not rect:continue
                    x0,y0,x1,y1=rect
                    source=[left+x0,top+y0,left+x1,top+y1]
                    x0,y0,x1,y1=source
                    apron=img.crop((x0-1,y0-1,x1+1,y1+1))
                    pieces.append((apron,source,depth))
        pieces.sort(key=lambda p:-p[0].height)
        x=y=row_height=0
        placements=[]
        for img,source,depth in pieces:
            if x+img.width>2048:
                x=0;y+=row_height;row_height=0
            placements.append((img,x,y,source,depth))
            x+=img.width;row_height=max(row_height,img.height)
        atlas=Image.new('RGBA',(2048,y+row_height))
        quads=[]
        for img,x,y,source,depth in placements:
            atlas.paste(img,(x,y))
            quads.append([round(depth,6),*source,x+1,y+1])
        base=input_file.stem
        atlas.save(directory/(base+'.webp'),quality=94,method=5)
        manifest=dict(size=atlas.size,quads=sorted(quads,key=lambda q:-q[0]))
        (directory/(base+'.json')).write_text(json.dumps(manifest,separators=(',',':')))
        print(base,atlas.size,len(quads),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('site',type=Path)
    args=parser.parse_args()
    with TemporaryDirectory() as temp:
        binary=Path(temp)/'splat-layers'
        subprocess.run(['c++','-std=c++17','-O3',Path(__file__).with_name('splat_layers.cpp'),'-o',binary],check=True)
        for file in sorted(args.site.glob('station-*.splatbin')):
            bake(binary,file,args.site)
