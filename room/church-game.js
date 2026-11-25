import * as THREE from 'three';
import {mergeGeometries} from 'three/addons/utils/BufferGeometryUtils.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';

// Game layout and geometry helpers use survey X, Y, elevation in metres.
const world = ([x,y,h]) => new THREE.Vector3(x,h,-y);

function windowLight(material, masonry) {
  material.customProgramCacheKey=()=>masonry?'stone-courses':'window-light';
  // Project the four rear window bays along the same direction as the shadow light.
  material.onBeforeCompile = shader => {
    shader.vertexShader = 'varying vec3 gameWorld;\n' + shader.vertexShader.replace(
      '#include <project_vertex>', '#include <project_vertex>\ngameWorld=(modelMatrix*vec4(transformed,1.)).xyz;');
    shader.fragmentShader = `varying vec3 gameWorld;
      float windowSun() {
        float travel = 22.9 + gameWorld.z;
        vec2 hit = vec2(gameWorld.x + .34*travel, gameWorld.y + .69*travel);
        float bay = mod(hit.x+2.83+1.615,3.23)-1.615;
        float side = .85-abs(bay);
        float arch = 2.40-.75*pow(abs(bay)/.85,2.)-hit.y;
        float opening = smoothstep(-.08,.10,min(min(side,arch),hit.y+1.95));
        opening *= smoothstep(.035,.105,abs(bay))*smoothstep(.04,.12,abs(hit.y-.12));
        return .18 + .82*opening;
      }
    ` + shader.fragmentShader.replace('#include <lights_fragment_begin>',
      THREE.ShaderChunk.lights_fragment_begin.replace('getDirectionalLightInfo( directionalLight, directLight );',
        'getDirectionalLightInfo( directionalLight, directLight ); directLight.color *= windowSun();'));
    if(masonry) shader.fragmentShader=shader.fragmentShader.replace('#include <color_fragment>', `
      #include <color_fragment>
      float course=gameWorld.y/.39;
      float bed=min(fract(course),1.-fract(course))*.39;
      float run=(gameWorld.x-gameWorld.z)/1.04+.47*mod(floor(course),2.);
      float joint=min(fract(run),1.-fract(run))*1.04;
      float mortar=1.-smoothstep(.003,.012,min(bed,joint));
      diffuseColor.rgb *= 1.-.21*mortar;
    `);
  };
}

function gameBuilder(scene, texture) {
  const batches = new Map();
  const materials = {};
  function material(name,color,options={}) {
    const mat = new THREE.MeshStandardMaterial({color,roughness:.89,...options});
    windowLight(mat,name==='stone');
    materials[name]=mat;
    batches.set(name,[]);
  }
  const limestone={map:texture,bumpMap:texture,bumpScale:.013};
  material('stone',0xc9c7c4,limestone);
  material('trim',0xe3ddcf,limestone);
  material('floor',0xbabdc5,limestone);
  material('joint',0x5b5249);
  material('oak',0x473126,{roughness:.67});
  material('oakTrim',0x624b37,{roughness:.60});
  material('iron',0x332c27,{roughness:.48,metalness:.32});
  material('glass',0xe5d8bf,{transparent:true,opacity:.34,depthWrite:false,roughness:.28});
  material('mantle',0xffe3ad,{emissive:0xffd9a0,emissiveIntensity:.9});
  material('board',0x302929);

  function add(name,geometry,matrix,variation=1) {
    const g=geometry.index ? geometry.toNonIndexed() : geometry;
    if(g!==geometry) geometry.dispose();
    if(matrix) g.applyMatrix4(matrix);
    const p=g.getAttribute('position'), n=g.getAttribute('normal');
    const uvs=[],colors=[];
    for(let i=0;i<p.count;i++){
      const x=p.getX(i),y=p.getY(i),z=p.getZ(i);
      const a=Math.abs(n.getX(i)),b=Math.abs(n.getY(i)),c=Math.abs(n.getZ(i));
      uvs.push((a>b&&a>c ? z:x)*.72,(b>a&&b>c ? z:y)*.72);
      colors.push(variation,variation,variation);
    }
    g.setAttribute('uv',new THREE.Float32BufferAttribute(uvs,2));
    g.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));
    batches.get(name).push(g);
  }
  function box(name,x,y,h,width,length,height,variation=1) {
    const matrix=new THREE.Matrix4().makeTranslation(x,h,-y);
    add(name,new THREE.BoxGeometry(width,height,length),matrix,variation);
  }
  function beam(name,a,b,width,depth=width) {
    const start=world(a),end=world(b),axis=end.clone().sub(start);
    const matrix=new THREE.Matrix4().compose(start.addScaledVector(axis,.5),
      new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0,1,0),axis.clone().normalize()),
      new THREE.Vector3(1,1,1));
    add(name,new THREE.BoxGeometry(width,axis.length(),depth),matrix);
  }
  function lathe(name,x,y,h,profile,sides) {
    const points=profile.map(([r,z])=>new THREE.Vector2(r,z));
    add(name,new THREE.LatheGeometry(points,sides),new THREE.Matrix4().makeTranslation(x,h,-y));
  }
  function faces(name,vertices,indices,shade=1) {
    const g=new THREE.BufferGeometry().setFromPoints(vertices.map(world));
    g.setIndex(indices);g.computeVertexNormals();add(name,g,null,shade);
  }
  function finish() {
    for(const [name,parts] of batches){
      if(!parts.length)continue;
      const mat=materials[name];mat.vertexColors=true;mat.side=THREE.DoubleSide;
      const mesh=new THREE.Mesh(mergeGeometries(parts),mat);
      parts.forEach(g=>g.dispose());
      mesh.castShadow=name!=='glass'&&name!=='mantle';mesh.receiveShadow=true;scene.add(mesh);
    }
  }
  return {add,box,beam,lathe,faces,finish};
}

function floorTiles(build,left,right,start,end,level) {
  build.box('joint',(left+right)/2,(start+end)/2,level-.10,right-left,end-start,.18);
  const rows=Math.ceil((end-start)/.70),step=(end-start)/rows;
  for(let row=0;row<rows;row++){
    const y=start+(row+.5)*step;
    const offset=(row%2)*.54;
    for(let at=left-offset;at<right;at+=1.08){
      const a=Math.max(left,at),b=Math.min(right,at+1.08);
      if(b-a<.015)continue;
      const variation=.84+.12*(.5+.5*Math.sin(83.17*at+13.2*y));
      build.box('floor',(a+b)/2,y,level-.044,b-a-.007,step-.007,.088,variation);
    }
  }
}

function stoneFlights(build,flights) {
  for(const flight of flights){
    for(let i=0;i<flight.length;i++){
      const q=flight[i];
      if(i%2===1){
        build.faces('stone',q,[0,1,2,0,2,3],.79);
        continue;
      }
      build.faces('floor',q,[0,1,2,0,2,3],.87+.08*Math.cos(i));
      // A narrow worn nosing on the downhill edge catches the window light.
      const lowEnd=flight[0][0][2]<flight.at(-1)[0][2]?[0,1]:[3,2];
      const a=q[lowEnd[0]],b=q[lowEnd[1]],d=lowEnd[0]===0?-.018:.018;
      const edge=[a,b,[b[0],b[1]+d,b[2]-.016],[a[0],a[1]+d,a[2]-.016]];
      build.faces('trim',edge,[0,1,2,0,2,3]);
    }
  }
}

function balustrade(build,x,start,end,z0,z1,base) {
  const level=y=>THREE.MathUtils.lerp(z0,z1,(y-start)/(end-start));
  const stones=Math.round((end-start)/.31);
  const step=(end-start)/stones;
  // The carved screen has real pointed openings through its full thickness.
  const shape=new THREE.Shape();
  shape.moveTo(0,base);shape.lineTo(end-start,base);shape.lineTo(end-start,z1+.94);
  shape.lineTo(0,z0+.94);shape.closePath();
  for(let i=0;i<stones;i++){
    const mid=(i+.5)*step, height=level(start+mid),half=step*.345;
    const hole=new THREE.Path();
    hole.moveTo(mid-half,height+.21);hole.lineTo(mid-half,height+.61);
    hole.quadraticCurveTo(mid-half,height+.73,mid,height+.83);
    hole.quadraticCurveTo(mid+half,height+.73,mid+half,height+.61);
    hole.lineTo(mid+half,height+.21);hole.closePath();shape.holes.push(hole);
  }
  const g=new THREE.ExtrudeGeometry(shape,{depth:.19,bevelEnabled:true,bevelThickness:.009,bevelSize:.009,bevelSegments:1,curveSegments:5});
  const matrix=new THREE.Matrix4().makeRotationY(Math.PI/2);
  matrix.setPosition(x-.095,0,-start);build.add('stone',g,matrix);
  function cap(offset,width,thickness){
    const a=[x,start,z0+offset],b=[x,end,z1+offset];
    // The column local X axis stays horizontal across the run.
    build.beam('trim',a,b,width,thickness);
  }
  cap(.975,.29,.095);cap(.915,.23,.055);cap(.17,.245,.067);
}

function lantern(build,x,y,h) {
  build.lathe('iron',x,y,h,[[.16,0],[.16,.04],[.11,.065],[.072,.12],[.06,.22],[.035,.28],[.038,.43],[.13,.50]],8);
  build.lathe('mantle',x,y,h,[[.022,.51],[.044,.55],[.04,.63],[.02,.68],[0,.70]],10);
  build.lathe('glass',x,y,h,[[.115,.50],[.16,.78]],6);
  build.lathe('iron',x,y,h,[[.165,.77],[.174,.80],[.12,.835],[.043,.93],[.025,.95],[.012,1.04],[0,1.06]],6);
  for(let i=0;i<6;i++){
    const a=i*Math.PI/3,s=Math.sin(a),c=Math.cos(a);
    build.beam('iron',[x+s*.114,y+c*.114,h+.49],[x+s*.161,y+c*.161,h+.78],.014);
  }
}
function post(build,x,y,h,light=true) {
  build.box('stone',x,y,h+.47,.36,.4,.94);
  build.box('trim',x,y,h+.92,.44,.47,.12);
  build.box('trim',x,y,h+1,.35,.39,.06);
  if(light)lantern(build,x,y,h+1.035);
}
function bench(build,x,y,h,width) {
  for(const dx of [-width*.39,width*.39]){
    build.box('oak',x+dx,y-.16,h+.22,.08,.075,.44);
    build.box('oak',x+dx,y+.17,h+.48,.085,.075,.96);
  }
  for(let i=0;i<3;i++)build.box('oakTrim',x,y-.15+i*.145,h+.46,width,.139,.065);
  for(const offset of [.67,.79,.87])build.box('oak',x,y+.18,h+offset,width,.054,.073);
  build.box('oakTrim',x,y-.215,h+.415,width,.043,.059);
  build.box('oakTrim',x,y+.18,h+.92,width+.055,.085,.054);
  build.box('oak',x,y,h+.22,width*.84,.07,.075);
}
// A recessed window has three successively narrower mouldings. The glazing visible
// through the opening comes from the photographed relief panel immediately behind it.
function windowOutline(center,half,sill,spring,crown) {
  const points=[[center-half,sill],[center-half,spring]];
  for(let i=1;i<=12;i++){
    const t=i/12;
    points.push([center-half+half*(1-Math.cos(t*Math.PI/2)),spring+(crown-spring)*Math.sin(t*Math.PI/2)]);
  }
  for(let i=1;i<=12;i++){
    const t=i/12;
    points.push([center+half*Math.sin(t*Math.PI/2),spring+(crown-spring)*Math.cos(t*Math.PI/2)]);
  }
  points.push([center+half,sill]);
  return points;
}
function northWindows(build) {
  const windows=[-2.82,.41,3.64,6.87];
  const face=21.52;
  const wall=new THREE.Shape();
  wall.moveTo(-4.62,-4.49);wall.lineTo(8.78,-4.49);wall.lineTo(8.78,5.84);wall.lineTo(-4.62,5.84);wall.closePath();
  for(const center of windows){
    const outline=windowOutline(center,1.02,-2.50,1.70,2.60);
    const path=new THREE.Path(outline.map(([x,y])=>new THREE.Vector2(x,y)));
    path.closePath();wall.holes.push(path);
    const profiles=[
      [1.12,-2.60,1.70,2.70,face-.08],
      [1.04,-2.52,1.70,2.62,face-.10],
      [1.02,-2.50,1.70,2.60,face+.01],
      [.84,-2.36,1.66,2.43,face+.39],
      [.80,-2.31,1.66,2.39,face+.42],
    ];
    let last;
    for(const [radius,sill,spring,crown,depth] of profiles){
      const ring=windowOutline(center,radius,sill,spring,crown).map(([x,h])=>[x,depth,h]);
      if(last){
        const indices=[];
        for(let i=0;i<ring.length;i++){
          const j=(i+1)%ring.length,n=ring.length;
          indices.push(i,j,n+j,i,n+j,n+i);
        }
        build.faces('trim',[...last,...ring],indices);
      }
      last=ring;
    }
    build.box('trim',center,face-.02,-2.57,2.23,.40,.11);
  }
  // The front plane sits inside the measured wall. Behind it are the deep photo-lined recesses.
  const masonry=new THREE.ShapeGeometry(wall,8);
  build.add('stone',masonry,new THREE.Matrix4().makeTranslation(0,0,-face));
  for(const x of [-4.43,2.05,8.53]){
    build.box('stone',x,face-.09,-.49,.38,.23,7.9);
    build.box('trim',x,face-.13,-4.17,.49,.31,.53);
    for(const dx of [-.16,.16])build.lathe('trim',x+dx,face-.20,-3.94,
      [[.075,0],[.07,6.78],[.12,6.83],[.12,6.93]],10);
  }
}
function fitOut(build){
  northWindows(build);
  const level=-4.435;
  floorTiles(build,-4.55,8.67,17.63,22.72,level);
  floorTiles(build,-4.55,2.18,7,13.01,-6.09);
  floorTiles(build,2.45,6.3,7,11.96,-2.28);
  floorTiles(build,-4.55,-1.1,12.98,17.64,-6.10);
  floorTiles(build,6.12,8.67,7,17.64,-4.45);
  // Stonework under the turnaround and elevated corridor.
  build.box('stone',2.04,17.69,-5.39,13.2,.13,1.90);
  build.box('stone',2.47,9.50,-4.2,.18,5,3.8);
  balustrade(build,-1.21,13.13,17.64,-6.12,-4.48,-6.22);
  for(const x of [2.28,6.32]){
    balustrade(build,x,11.98,17.65,-2.24,-4.47,-6.21);
    balustrade(build,x,7.25,11.98,-2.24,-2.24,-6.21);
    post(build,x,11.98,-2.28);
  }
  for(const x of [-1.21,2.28,6.32])post(build,x,17.68,level);
  for(const x of [-2,1.06,4.55,7.01])bench(build,x,21.15,level,2.05);
  const sx=-.95,sy=19.55;
  for(const side of [-1,1]){
    build.beam('oakTrim',[sx+side*.44,sy-.27,level],[sx+side*.28,sy,level+1.57],.048);
  }
  build.beam('oak',[sx,sy+.48,level],[sx,sy,level+1.38],.055);
  build.box('oakTrim',sx,sy,level+1.18,.84,.065,.86);
  build.box('board',sx,sy-.04,level+1.18,.775,.025,.795);
  // An authored return wall closes the demonstration at the south end.
  build.box('stone',2.05,6.98,-.45,13.9,.30,12);
  for(const x of [.32,4.37]){
    const h=x<1?-6.09:-2.28;
    build.box('oak',x,7.17,h+1.37,2.08,.15,2.74);
    for(let i=0;i<12;i++)build.box('oakTrim',x+(i-5.5)*.166,7.26,h+1.35,.158,.035,2.66,.70+.2*Math.cos(i));
    for(const dx of [-1.13,1.13])build.box('trim',x+dx,7.22,h+1.45,.18,.24,2.90);
    build.box('trim',x,7.22,h+2.87,2.46,.25,.19);
  }
}

export async function createChurchGame(renderer) {
  const [enclosure,texture,flights]=await Promise.all([
    new GLTFLoader().loadAsync('./game-enclosure.glb'),
    new THREE.TextureLoader().loadAsync('./game-limestone.webp'),
    fetch('./game-flights.json').then(r=>{if(!r.ok)throw Error('樓梯量測未能載入');return r.json();})
  ]);
  texture.wrapS=texture.wrapT=THREE.RepeatWrapping;texture.colorSpace=THREE.SRGBColorSpace;
  texture.anisotropy=Math.min(8,renderer.capabilities.getMaxAnisotropy());
  const scene=new THREE.Scene();scene.background=new THREE.Color(0x756c62);
  scene.add(enclosure.scene);
  const build=gameBuilder(scene,texture);stoneFlights(build,flights);fitOut(build);build.finish();
  scene.add(new THREE.HemisphereLight(0xe4edff,0xcabdb5,1.43));
  const sun=new THREE.DirectionalLight(0xfff2df,3.1);
  sun.target.position.set(2,-4,-16);sun.position.copy(sun.target.position).add(new THREE.Vector3(3.4,6.9,-10));
  sun.castShadow=true;
  const c=sun.shadow.camera;c.left=-10;c.right=10;c.bottom=-9;c.top=9;c.near=.4;c.far=38;
  sun.shadow.mapSize.setScalar(1024);sun.shadow.normalBias=.011;sun.shadow.bias=-.00025;
  scene.add(sun,sun.target);
  return scene;
}

// Walkable rectangle unions leave a margin beside each modeled railing.
export function inChurchGame(x,z) {
  const y=-z;
  return (y>=17.88&&y<=20.68&&x>=-3.98&&x<=8.12)
    || (y>=9.4&&y<17.9&&x>=-.91&&x<=1.92)
    || (y>=9.4&&y<17.9&&x>=2.63&&x<=5.95);
}
