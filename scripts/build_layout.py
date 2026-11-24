"""Build the independently editable, photo-guided architectural interpretation.
Run with blender -b -P scripts/build_layout.py -- room/site
"""
import math
import sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector

OUT=Path(sys.argv[sys.argv.index('--')+1]).resolve()
bpy.ops.wm.read_factory_settings(use_empty=True)
def linear(v):
    v=v/255
    return v/12.92 if v<.04045 else ((v+.055)/1.055)**2.4
def mat(name,hexcolor,metal=0,rough=.68):
    rgb=bytes.fromhex(hexcolor)
    m=bpy.data.materials.new(name);m.use_nodes=True
    bs=m.node_tree.nodes.get('Principled BSDF')
    bs.inputs['Base Color'].default_value=(*[linear(n) for n in rgb],1)
    bs.inputs['Roughness'].default_value=rough;bs.inputs['Metallic'].default_value=metal
    return m
walnut=mat('French walnut','633e2e');edge=mat('Polished dark wood','432b24');inlay=mat('Walnut inlay','8b5e43')
brass=mat('Aged brass','b38a51',.5,.4);wall=mat('Rose plaster','af8073');ivory=mat('Ivory painted timber','e4d9c5')
linen=mat('Warm linen','efe3ce');blush=mat('Woven rose velvet','a06b59');jade=mat('Olive velvet','958c66')
rug=mat('Madder wool','874e40');ochre=mat('Woven ochre','c19466');sand=mat('Wool carpet','c5b6a0')
glass=mat('Daylit glazing','cadbd4');mirror=mat('Silvered mirror','9eafa8',.35,.3)
stone=mat('Fireplace limestone','b5a78e');coal=mat('Firebox iron','352f2b');green=mat('Landscape green','7c8972')
def loc(x,h,z):return (x,-z,h)
def group(name,x=0,h=0,z=0,yaw=0,parent=None):
    obj=bpy.data.objects.new(name,None);bpy.context.collection.objects.link(obj)
    obj.parent=parent;obj.location=loc(x,h,z);obj.rotation_euler.z=yaw;return obj
def finish(obj,name,m,parent):
    obj.name=name;obj.data.materials.append(m);obj.parent=parent;return obj
def box(parent,name,x,h,z,w,H,d,m,r=.012):
    bpy.ops.mesh.primitive_cube_add(size=1,location=loc(x,h,z));o=bpy.context.object
    o.dimensions=(w,d,H);bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    if r:
        b=o.modifiers.new('Soft edges','BEVEL');b.width=min(r,min(w,H,d)*.49);b.segments=3
        n=o.modifiers.new('Corner normals','WEIGHTED_NORMAL')
    return finish(o,name,m,parent)
def ball(parent,name,x,h,z,w,H,d,m):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20,ring_count=12,location=loc(x,h,z))
    o=bpy.context.object;o.scale=(w*.5,d*.5,H*.5)
    for f in o.data.polygons:f.use_smooth=True
    return finish(o,name,m,parent)
def cone(parent,name,x,h,z,r1,r2,H,m):
    bpy.ops.mesh.primitive_cone_add(vertices=24,radius1=r1,radius2=r2,depth=H,location=loc(x,h,z))
    o=bpy.context.object
    for f in o.data.polygons:
        if len(f.vertices)==4:f.use_smooth=True
    return finish(o,name,m,parent)
def leg(p,x,z,H=.64):
    cone(p,'Turned leg',x,H/2,z,.029,.044,H,walnut)
    for level in (.11,H*.82):ball(p,'Collar',x,level,z,.085,.055,.085,edge)
def drawer(p,x,h,z,w,H):
    box(p,'Drawer face',x,h,z,w,H,.035,inlay)
    box(p,'Recessed panel',x,h,z-.022,w-.075,H-.045,.015,walnut,.005)
    ball(p,'Brass knob',x,h,z-.048,.043,.042,.035,brass)
def lamp(p,x,h,z):
    cone(p,'Lamp foot',x,h+.018,z,.10,.085,.036,brass)
    ball(p,'Ceramic lamp',x,h+.14,z,.14,.22,.14,ivory)
    cone(p,'Socket',x,h+.28,z,.028,.026,.08,brass)
    cone(p,'Linen lampshade',x,h+.40,z,.145,.086,.22,linen)
def rectbox(p,name,x,z,w,d,m,h=.016):
    return box(p,name,x,h,z,w,.008,d,m,.001)
def textile():
    # Rectified fabric patch from the bed photograph; reflected tiling avoids a hard seam.
    photo=bpy.data.images.load(str(OUT/'frame-9637.jpg'));w,h=photo.size
    pixels=np.empty(w*h*4,'f4');photo.pixels.foreach_get(pixels)
    img=pixels.reshape(h,w,4);tile=img[h-328:h-231,220:349].copy()
    tile=np.concatenate([tile,tile[:,::-1]],axis=1);tile=np.concatenate([tile,tile[::-1]],axis=0)
    im=bpy.data.images.new('Bedspread fabric sample',width=tile.shape[1],height=tile.shape[0],alpha=False)
    im.pixels.foreach_set(tile.ravel());im.file_format='PNG';im.pack()
    m=mat('Photographed bedspread','ac8468');tree=m.node_tree
    tex=tree.nodes.new('ShaderNodeTexImage');tex.image=im
    uv=tree.nodes.new('ShaderNodeTexCoord');mapping=tree.nodes.new('ShaderNodeMapping')
    mapping.inputs['Scale'].default_value=(5,6,1)
    tree.links.new(uv.outputs['UV'],mapping.inputs[0]);tree.links.new(mapping.outputs[0],tex.inputs[0]);tree.links.new(tex.outputs[0],tree.nodes.get('Principled BSDF').inputs['Base Color'])
    return m
fabric=textile()
root=group('Bedroom layout')
x0,x1,z0,z1=-2.35,3.38,-2.44,2.33
W=x1-x0;D=z1-z0
floor=group('Carpet and furniture',parent=root)
box(floor,'Floor structure',.515,-.074,-.055,W+.09,.14,D+.09,edge,.025)
box(floor,'Wool carpet',.515,-.006,-.055,W,.022,D,sand)
# Wall groups can be removed independently for the viewer cutaway and in editors.
def wall_group(name,x,z,yaw,length):
    p=group(name,x,0,z,yaw,root)
    box(p,'Plaster',0,1.73,0,length,3.46,.085,wall,.005)
    box(p,'Timber dado',0,.435,-.055,length,.87,.035,walnut,.003)
    box(p,'Skirting',0,.07,-.081,length,.14,.065,edge)
    box(p,'Dado rail',0,.91,-.08,length,.07,.075,inlay)
    count=max(1,round(length/.65));span=length/count
    for i in range(count):
        cx=-length/2+span*(i+.5)
        box(p,'Wainscot field',cx,.48,-.076,span-.086,.56,.016,inlay,.005)
        box(p,'Recessed wood',cx,.48,-.087,span-.112,.535,.010,walnut,.004)
    return p
back=wall_group('Shell_back',.515,z1,0,W)
front=wall_group('Shell_front',.515,z0,math.pi,W)
left=wall_group('Shell_left',x0,-.055,-math.pi/2,D)
right=wall_group('Shell_right',x1,-.055,math.pi/2,D)
cap=group('Shell_ceiling',parent=root)
box(cap,'Ceiling',.515,3.47,-.055,W,.075,D,ivory)
def curtain(p,cx):
    # Soft continuous fluted surface for a gathered curtain.
    nx,ny=32,9;verts=[];faces=[]
    for j in range(ny+1):
        t=j/ny
        for i in range(nx+1):
            u=i/nx; xx=cx+(u-.5)*(.17+.10*t)
            hh=2.43-1.63*t+.016*math.cos(5*math.pi*u)*t**5
            zz=-.18-.038*math.cos(8*math.pi*u)
            verts.append(loc(xx,hh,zz))
    for j in range(ny):
        for i in range(nx):
            a=j*(nx+1)+i;faces.append((a,a+1,a+nx+2,a+nx+1))
    mesh=bpy.data.meshes.new('Linen pleats');mesh.from_pydata(verts,[],faces)
    o=bpy.data.objects.new('Pleated curtain',mesh);bpy.context.collection.objects.link(o);finish(o,'Pleated curtain',linen,p)
    for face in mesh.polygons:face.use_smooth=True
for cx in (-1.12,1.63):
    box(back,'Window casing',cx,1.67,-.075,.88,1.62,.065,ivory)
    box(back,'Glass',cx,1.67,-.115,.76,1.5,.035,glass)
    box(back,'Sash',cx,1.67,-.14,.032,1.52,.027,ivory)
    box(back,'Sash',cx,1.67,-.14,.79,.035,.027,ivory)
    box(back,'Sill',cx,.86,-.14,.98,.065,.19,ivory)
    box(back,'Curtain rod',cx,2.46,-.14,1.1,.026,.032,brass)
    curtain(back,cx-.38);curtain(back,cx+.38)
# Door, picture frames and an inset architectural hearth.
box(front,'Door architrave',-.15,1.05,-.083,.98,2.10,.08,ivory)
box(front,'Paneled door',-.15,1.01,-.133,.84,2.01,.045,inlay)
for hh,H in ((.47,.65),(1.39,.89)):box(front,'Raised door panel',-.15,hh,-.163,.65,H,.02,walnut)
ball(front,'Handle',.17,.99,-.19,.065,.065,.06,brass)
def picture(p,cx,hh,w,H):
    box(p,'Picture frame',cx,hh,-.088,w,H,.073,brass)
    box(p,'Mat board',cx,hh,-.131,w-.05,H-.05,.015,linen)
    box(p,'Landscape print',cx,hh,-.145,w-.13,H-.14,.009,green)
picture(left,-1.34,1.65,.40,.53);picture(right,-1.15,1.71,.50,.65)
fire=group('Fireplace',x0+.15,0,.06,-math.pi/2,floor)
box(fire,'Hearth',0,.04,-.13,1.58,.09,.55,stone)
box(fire,'Firebox',0,.55,-.025,1.20,.96,.15,coal)
for xx in (-.61,.61):
    box(fire,'Fireplace column',xx,.56,-.13,.22,1.04,.27,walnut)
    box(fire,'Column trim',xx,.57,-.273,.14,.78,.032,inlay)
box(fire,'Lintel',0,1.04,-.15,1.46,.21,.30,walnut)
box(fire,'Mantel',0,1.2,-.16,1.63,.09,.41,inlay)
for xx in (-.34,-.18,0,.18,.34):cone(fire,'Grate bar',xx,.23,-.21,.017,.017,.31,brass)
box(fire,'Grate rail',0,.36,-.21,.79,.033,.04,brass)
for xx in (-.4,.41):ball(fire,'Mantel pottery',xx,1.35,-.15,.12,.22,.12,ivory)
# Bed points toward the windows on the back wall.
bed=group('Bed',.68,0,.97,0,floor)
for xx in (-.73,.73):
    for zz in (-1.01,1.01):
        leg(bed,xx,zz,.38)
        H=1.19 if zz>0 else .76
        cone(bed,'Bedpost',xx,H*.5,zz,.048,.052,H,edge)
        ball(bed,'Finial',xx,H+.04,zz,.115,.13,.115,inlay)
box(bed,'Bed rails',0,.32,0,1.55,.23,2.12,walnut,.05)
box(bed,'Mattress',0,.49,0,1.49,.25,2.04,linen,.105)
box(bed,'Quilt',0,.63,-.17,1.54,.17,1.80,fabric,.078)
box(bed,'Folded coverlet',0,.722,-.76,1.52,.045,.34,blush,.02)
for zz,hh,H in ((1.009,.77,.70),(-1.01,.48,.35)):
    box(bed,'Carved end panel',0,hh,zz,1.48,H,.075,walnut,.036)
    ball(bed,'Arched crown',0,hh+H*.40,zz,1.35,H*.44,.078,inlay)
for xx in (-.36,.36):
    p=box(bed,'Linen pillow',xx,.76,.70,.72,.16,.42,linen,.078);p.rotation_euler.x=-.18
    p=box(bed,'Tapestry cushion',xx,.78,.43,.62,.19,.32,fabric,.086);p.rotation_euler.x=-.32
for pos in (-.43,1.78):
    t=group('Bedside table',pos,0,1.78,0,floor)
    for xx in (-.19,.19):
        for zz in (-.16,.16):leg(t,xx,zz,.56)
    box(t,'Apron',0,.52,0,.45,.17,.39,walnut);drawer(t,0,.53,-.197,.4,.13)
    box(t,'Top',0,.64,0,.51,.055,.45,inlay);lamp(t,0,.67,0)
# Rectify the photographed rug from its four visible corners.
photo=bpy.data.images.load(str(OUT/'frame-17521.jpg'))
w,h=photo.size
src=np.empty(w*h*4,'f4');photo.pixels.foreach_get(src);src=src.reshape(h,w,4)[::-1]
matrix=[];target=[]
for (u,v),(x,y) in zip([(0,0),(1,0),(1,1),(0,1)],[(5,150),(200,79),(446,195),(260,362)]):
    matrix.extend([(u,v,1,0,0,0,-u*x,-v*x),(0,0,0,u,v,1,-u*y,-v*y)])
    target.extend([x,y])
projection=np.r_[np.linalg.solve(matrix,target),1].reshape(3,3)
u,v=np.meshgrid(np.linspace(0,1,384),np.linspace(0,1,384))
rays=np.stack([u,v,np.ones_like(u)],axis=-1)@projection.T
xx,yy=(rays[:,:,:2]/rays[:,:,2:]).transpose(2,0,1)
ix,iy=xx.astype(int),yy.astype(int);fx=(xx-ix)[:,:,None];fy=(yy-iy)[:,:,None]
rgba=(1-fy)*((1-fx)*src[iy,ix]+fx*src[iy,ix+1])+fy*((1-fx)*src[iy+1,ix]+fx*src[iy+1,ix+1])
rgba[:,:,:3]=np.minimum(rgba[:,:,:3]*4.0,1)
im=bpy.data.images.new('Photographed wool rug',width=384,height=384,alpha=False)
im.pixels.foreach_set(np.ascontiguousarray(rgba[::-1],dtype='f4').ravel());im.pack()
m=mat('Rug photograph','874e40');tree=m.node_tree;tex=tree.nodes.new('ShaderNodeTexImage');tex.image=im
tree.links.new(tex.outputs[0],tree.nodes.get('Principled BSDF').inputs['Base Color'])
rectbox(floor,'Bound rug edge',.66,-.66,1.55,1.46,rug,.016)
bpy.ops.mesh.primitive_plane_add(size=1,location=loc(.66,.022,-.66));obj=bpy.context.object
obj.scale=(1.52,1.43,1);finish(obj,'Photographed rug face',m,floor)
def chair(name,x,z,yaw,fabricmat,arms):
    p=group(name,x,0,z,yaw,floor)
    for xx in (-.22,.22):
        for zz in (-.20,.20):leg(p,xx,zz,.45)
    box(p,'Chair seat frame',0,.42,0,.55,.10,.51,walnut,.04)
    box(p,'Upholstered seat',0,.50,0,.52,.12,.48,fabricmat,.053)
    for xx in (-.225,.225):cone(p,'Back upright',xx,.78,.20,.026,.028,.62,walnut)
    ball(p,'Oval back frame',0,.90,.21,.51,.53,.095,walnut)
    ball(p,'Upholstered back',0,.90,.176,.42,.43,.075,fabricmat)
    if arms:
        for xx in (-.275,.275):
            cone(p,'Arm support',xx,.58,-.12,.018,.02,.23,walnut)
            box(p,'Armrest',xx,.70,.01,.073,.055,.41,inlay,.025)
chair('Window armchair',-1.38,1.57,-.56,jade,True)
chair('Side chair',2.25,1.00,.62,blush,False)
desk=group('Writing desk',-1.42,0,-1.65,-2.48,floor)
for xx in (-.54,.54):
    for zz in (-.26,.26):leg(desk,xx,zz,.73)
box(desk,'Writing surface',0,.74,0,1.30,.085,.76,inlay)
box(desk,'Desk gallery',0,1.01,.29,1.25,.48,.075,walnut)
for xx in (-.59,-.20,.20,.59):box(desk,'Pigeonhole divider',xx,.95,.18,.04,.30,.25,inlay)
box(desk,'Desk ledge',0,1.22,.19,1.30,.055,.35,inlay)
for xx in (-.34,.34):drawer(desk,xx,.66,-.34,.57,.12)
box(desk,'Stationery',-.22,.787,-.15,.25,.008,.18,linen,.002)
lamp(desk,.43,.79,.02)
chair('Writing chair',-.83,-1.05,.65,blush,False)
van=group('Mirror and dresser',2.05,0,-2.13,math.pi,floor)
box(van,'Dresser cabinet',0,.47,0,1.04,.80,.45,walnut,.035)
for hh in (.26,.48,.70):drawer(van,0,hh,-.226,.92,.18)
box(van,'Dresser top',0,.91,0,1.14,.067,.52,inlay)
box(van,'Mirror frame',0,1.48,.12,.82,1.08,.086,walnut,.037)
box(van,'Mirror inset',0,1.48,.062,.71,.95,.017,mirror,.011)
ball(van,'Mirror crown',0,2.01,.12,.82,.21,.09,inlay)
# Batch static geometry by material inside each editable furniture/wall group.
for parent in [obj for obj in bpy.context.scene.objects if obj.type=='EMPTY']:
    pieces=[ob for ob in parent.children if ob.type=='MESH']
    if not pieces:continue
    buckets={}
    for ob in pieces:buckets.setdefault(ob.data.materials[0],[]).append(ob)
    for material,objs in buckets.items():
        bpy.ops.object.select_all(action='DESELECT')
        for ob in objs:ob.select_set(True)
        bpy.context.view_layer.objects.active=objs[0];bpy.ops.object.convert(target='MESH')
        if len(objs)>1:bpy.ops.object.join()
        bpy.context.view_layer.objects.active.name=parent.name+' · '+material.name
bpy.ops.export_scene.gltf(filepath=str(OUT/'bedroom-layout.glb'),export_format='GLB',export_yup=True,export_extras=False)
