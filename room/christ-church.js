import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';

import {PhotoWalk} from './church-photowalk.js';
import {createChurchGame, inChurchGame} from './church-game.js';

const $ = id => document.getElementById(id);
const canvas = $('view');
const turn = new THREE.Euler(0, 0, 0, 'YXZ');
const camera = new THREE.PerspectiveCamera(70, 1, .055, 165);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x302e28);
const keys = new Set();
const moveKeys = {ArrowUp:'KeyW', ArrowDown:'KeyS', ArrowLeft:'KeyA', ArrowRight:'KeyD'};
const tmp = new THREE.Vector3();
const cut = new THREE.Plane(new THREE.Vector3(0,-1,0), 2.8);
const uniform = {
  churchLift:{value:0}, churchPhoto:{value:null}, churchDepth:{value:null}, churchInverse:{value:new THREE.Matrix4()},
  churchLens:{value:new THREE.Vector4()}, churchDist:{value:new THREE.Vector4()}, churchGain:{value:0}
};
let renderer, controls, views, grid, current, model, mode='walk', dirty=true, ready=false;
let detailView, pending, generation=0, lastTime=0, lastSelection=0, drag, benchmark;
let photoWalk, photoWalkLoad, photoWalkRetry=0;
let gameScene, scanLoad, gameLoad, action=0, photoReturn;
const strolling=()=>mode==='walk'||mode==='game';
const photoStops=[197,199,200,201,202,204,205,206,207,208,209,210];
let stopIndex=0,routeOpen=false;
const cache = new Map();
const textureLoader = new THREE.TextureLoader();

function photoProjection(material) {
  material.onBeforeCompile = shader => {
    Object.assign(shader.uniforms, uniform);
    shader.vertexShader = 'varying vec3 churchPos;\n'+shader.vertexShader.replace('#include <project_vertex>',
      '#include <project_vertex>\nchurchPos=(modelMatrix*vec4(transformed,1.)).xyz;');
    shader.fragmentShader = `
      varying vec3 churchPos;
      uniform mat4 churchInverse;
      uniform vec4 churchLens,churchDist;
      uniform sampler2D churchPhoto,churchDepth;
      uniform float churchGain,churchLift;
    `+shader.fragmentShader.replace('#include <color_fragment>', `
      #include <color_fragment>
      if(churchGain>0.001){
        vec3 p=(churchInverse*vec4(churchPos,1.)).xyz;
        float r=length(p.xy);
        float theta=atan(r,-p.z);
        float t=theta*theta;
        float radial=theta*(1.+t*(churchDist.x+t*(churchDist.y+t*(churchDist.z+t*churchDist.w))));
        vec2 uv=(vec2(p.x,-p.y)*radial/max(r,.00001)*churchLens.xy+churchLens.zw+0.5)/vec2(1440.,1080.);
        if(p.z<0. && all(greaterThan(uv,vec2(0.))) && all(lessThan(uv,vec2(1.)))){
          vec2 bytes=floor(texture2D(churchDepth,uv).rg*255.+.5);
          float depth=dot(bytes,vec2(256.,1.))*.001;
          float visible=1.-smoothstep(.13+.012*depth,.3+.023*depth,abs(depth-length(p)));
          float edge=clamp(35.*min(min(uv.x,1.-uv.x),min(uv.y,1.-uv.y)),0.,1.);
          diffuseColor.rgb=mix(diffuseColor.rgb,texture2D(churchPhoto,uv).rgb,churchGain*edge*visible);
        }
      }
      if(churchLift>.5) diffuseColor.rgb=pow(max(diffuseColor.rgb,vec3(0.)),vec3(.80));
    `);
  };
}

async function fetchDetail(view) {
  if(cache.has(view.id)) return cache.get(view.id);
  const pair=await Promise.allSettled([textureLoader.loadAsync('photo-'+view.id+'.webp'),textureLoader.loadAsync('depth-'+view.id+'.png?v=railrepair1')]);
  if(pair.some(result=>result.status==='rejected')){
    for(const result of pair) if(result.status==='fulfilled') result.value.dispose();
    throw new Error('照片讀取失敗');
  }
  const [photo,depth]=pair.map(result=>result.value);
  photo.flipY=false;
  photo.colorSpace=THREE.SRGBColorSpace;
  depth.flipY=false;
  depth.generateMipmaps=false;
  depth.minFilter=depth.magFilter=THREE.NearestFilter;
  const result={photo,depth};
  cache.set(view.id,result);
  return result;
}

function releaseCache(keep) {
  for(const [id,pair] of cache){
    if(cache.size<=5) break;
    if(id===keep || id===detailView?.id) continue;
    pair.photo.dispose(); pair.depth.dispose(); cache.delete(id);
  }
}

function selectPhotoWalk(){
  const {x,z}=camera.position;
  const requested=mode==='walk' && $('detail').checked && z < -8 && z > -22 && x > -5 && x < 9;
  if(photoWalk)photoWalk.enabled=requested;
  if(!requested){canvas.dataset.photowalk='off';return;}
  if(!photoWalk){
    if(!photoWalkLoad && performance.now()>=photoWalkRetry){
      photoWalkLoad=PhotoWalk.create(views,()=>{dirty=true;}).then(walk=>{
        photoWalk=walk;selectPhotoWalk();dirty=true;
      }).catch(()=>{photoWalkRetry=performance.now()+3000;}).finally(()=>{photoWalkLoad=null;});
    }
    canvas.dataset.photowalk='loading';return;
  }
  photoWalk.select(camera);
  canvas.dataset.photowalk=photoWalk.active.length?'ready':'loading';
}

function selectDetail(){
  selectPhotoWalk();
  if(mode!=='walk'||!$('detail').checked||(photoWalk?.enabled&&photoWalk.active.length)){uniform.churchGain.value=0;return;}
  const direction=camera.getWorldDirection(tmp);
  let best,lowest=Infinity;
  for(const view of views){
    if(view.detail===false) continue;
    const score=camera.position.distanceToSquared(view.position)+4.2*(1-direction.dot(view.forward));
    if(score<lowest){lowest=score;best=view;}
  }
  const gain=detailView ? 1-THREE.MathUtils.smoothstep(camera.position.distanceTo(detailView.position),.20,2.8) : 0;
  if(Math.abs(uniform.churchGain.value-gain)>.001){uniform.churchGain.value=gain;dirty=true;}
  if(lowest>9 || best===detailView || best===pending) return;
  const token=++generation;
  pending=best;
  fetchDetail(best).then(({photo,depth})=>{
    if(token!==generation){releaseCache(detailView?.id);return;}
    pending=null; detailView=best;
    uniform.churchPhoto.value=photo; uniform.churchDepth.value=depth;
    uniform.churchInverse.value.copy(best.matrix).invert();
    uniform.churchLens.value.fromArray(best.lens);uniform.churchDist.value.fromArray(best.distortion);
    releaseCache(best.id);selectDetail();dirty=true;
  }).catch(()=>{if(token===generation) pending=null;});
}

function setMode(next) {
  mode=next;
  if(photoWalk)photoWalk.enabled=false;
  canvas.dataset.photowalk='off';
  benchmark=null;
  uniform.churchLift.value=next==='overview'?1:0;
  camera.fov=next==='overview'?38:70;
  camera.updateProjectionMatrix();
  scene.background.setHex(next==='overview'?0xe2ddd1:0x302e28);
  keys.clear();
  controls.enabled=next==='overview';
  $('pad').hidden=!strolling();
  $('photo-stops').hidden=next!=='walk'||!routeOpen;
  $('reference').hidden=next!=='compare';
  $('detail').disabled=next!=='walk';
  $('bench').disabled=next==='compare';
  for(const id of ['walk','game','overview','compare']) $(id).setAttribute('aria-pressed',String(id===next));
  for(const material of model.materials) material.clippingPlanes=next==='overview'?[cut]:[];
  uniform.churchGain.value=0;
  $('help').textContent=next==='game'?'沿石階走到平台，可轉身走上另一段樓梯。石材、欄杆與燈具為遊戲美術重繪。':next==='overview'?'拖曳旋轉、滾輪或雙指縮放。為看清配置，這個視圖切開屋頂。':next==='compare'?'原始魚眼相機照片，可檢查窗框、桌沿和石階的細節。':'拖曳環顧，用方向鍵或 WASD 沿拍攝區域移動。';
  dirty=true;
}

function positionAt(view,lookUp=false) {
  current=view;
  camera.position.copy(view.position);
  turn.setFromRotationMatrix(view.matrix,'YXZ');turn.z=0;
  if(view.id===198||lookUp) turn.x+=.24;
  camera.quaternion.setFromEuler(turn);
  $('caption').textContent=view.name||'鄰近拍攝點';
  highlight(view);
}

function station(view) {
  routeOpen=false;setMode('walk');positionAt(view);
  selectDetail();
  canvas.focus({preventScroll:true});
}

function highlight(view){
  for(const button of $('stations').children) button.setAttribute('aria-pressed',String(+button.dataset.id===view?.id));
}

function overview(){
  setMode('overview');highlight(null);
  controls.target.set(2,-.5,8);
  const offset=(camera.aspect>1 ? new THREE.Vector3(1,.9,.40) : new THREE.Vector3(.24,1.4,1)).normalize();
  const rotation=new THREE.Matrix4().lookAt(offset,new THREE.Vector3(),camera.up);
  const inverse=new THREE.Quaternion().setFromRotationMatrix(rotation).invert();
  const halfHeight=Math.tan(THREE.MathUtils.degToRad(camera.fov/2));
  let distance=0;
  for(const x of [-7,13])for(const y of [-5,3])for(const z of [-23,40]){
    const p=new THREE.Vector3(x,y,z).sub(controls.target).applyQuaternion(inverse);
    distance=Math.max(distance,p.z+Math.max(Math.abs(p.y),Math.abs(p.x)/camera.aspect)/halfHeight);
  }
  camera.position.copy(controls.target).addScaledVector(offset,distance*1.02);
  camera.lookAt(controls.target);controls.update();
  $('caption').textContent='大廳與樓梯 · 全區剖視';
}

function constrain(x,z){
  // Test intermediate cells too, so a slow frame cannot jump across a wall.
  const dx=x-camera.position.x, dz=z-camera.position.z;
  const steps=Math.max(1,Math.ceil(Math.hypot(dx,dz)/(grid.cell*.5)));
  let height=camera.position.y;
  for(let i=1;i<=steps;i++){
    if(mode==='game'&&!inChurchGame(camera.position.x+dx*i/steps,camera.position.z+dz*i/steps)) return;
    const col=Math.round((camera.position.x+dx*i/steps-grid.origin[0])/grid.cell);
    const row=Math.round((-camera.position.z-dz*i/steps-grid.origin[1])/grid.cell);
    if(col<0||row<0||col>=grid.cols||row>=grid.rows) return;
    const cm=grid.values[row*grid.cols+col];
    if(cm===-32768 || Math.abs(cm*.01-height)>.4) return;
    height=cm*.01;
  }
  camera.position.set(x,height,z);
  $('caption').textContent=mode==='game'?'遊戲重繪 · 石階':routeOpen?'自由移動 · 合成視角':'室內漫遊';highlight(null);
}

function step(time){
  requestAnimationFrame(step);
  if(!ready||document.hidden){lastTime=0;return;}
  const dt=lastTime?Math.min(.1,(time-lastTime)/1000):0;
  lastTime=time;
  if(strolling()&&keys.size){
    const forward=Number(keys.has('KeyW'))-Number(keys.has('KeyS'));
    const right=Number(keys.has('KeyD'))-Number(keys.has('KeyA'));
    const speed=dt*2.35/Math.max(1,Math.hypot(forward,right));
    const c=Math.cos(turn.y),s=Math.sin(turn.y);
    constrain(camera.position.x+speed*(-s*forward+c*right),camera.position.z+speed*(-c*forward-s*right));
    dirty=true;
  }
  if(mode==='overview') controls.update();
  if(benchmark){
    if(benchmark.last) benchmark.frames.push(time-benchmark.last);
    benchmark.last=time;
    if(strolling()){turn.y+=dt*.13;camera.quaternion.setFromEuler(turn);}
    else {camera.position.sub(controls.target).applyAxisAngle(THREE.Object3D.DEFAULT_UP,dt*.13).add(controls.target);camera.lookAt(controls.target);}
    dirty=true;
    if(time-benchmark.start>4500 && benchmark.frames.length){
      const frames=benchmark.frames,avg=1000*frames.length/frames.reduce((a,b)=>a+b,0);
      frames.sort((a,b)=>a-b);
      $('help').textContent=`本機平均 ${avg.toFixed(1)} FPS · P95 ${frames[Math.min(frames.length-1,Math.floor(frames.length*.95))].toFixed(1)} ms · ${canvas.width} × ${canvas.height}`;
      benchmark=null;$('bench').disabled=false;
    }
  }
  if(time-lastSelection>140){selectDetail();lastSelection=time;}
  if(dirty&&mode!=='compare'){
    if(mode==='walk'&&photoWalk?.enabled&&photoWalk.active.length){
      uniform.churchGain.value=0;photoWalk.render(renderer,camera,scene);
    }else renderer.render(mode==='game'?gameScene:scene,camera);
    dirty=false;
  }
}

async function start(){
  renderer=new THREE.WebGLRenderer({canvas,antialias:true,powerPreference:'high-performance'});
  renderer.localClippingEnabled=true;
  renderer.shadowMap.enabled=true;
  renderer.shadowMap.type=THREE.PCFShadowMap;
  // Static game geometry and lighting share a single shadow map across camera poses.
  renderer.shadowMap.autoUpdate=false;
  renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));
  controls=new OrbitControls(camera,canvas);controls.enabled=false;controls.minDistance=5;controls.maxDistance=115;
  controls.maxPolarAngle=Math.PI*.485;
  controls.addEventListener('change',()=>{dirty=true;});
  const resize=()=>{
    const w=canvas.clientWidth,h=canvas.clientHeight;
    renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();dirty=true;
  };
  new ResizeObserver(resize).observe(canvas);resize();
  const compact=window.matchMedia('(max-width: 680px)').matches;
  const modelFile=compact?'christ-church-mobile.glb':'christ-church.glb';
  const data=await fetch('./scene.json?v=railrepair1').then(r=>{if(!r.ok)throw Error();return r.json();});
  views=data.views;
  grid=data.walkGrid;
  const bytes=Uint8Array.from(atob(grid.heights),c=>c.charCodeAt(0));
  const heights=new DataView(bytes.buffer);
  grid.values=Int16Array.from({length:grid.cols*grid.rows},(_,i)=>heights.getInt16(i*2,true));
  for(const view of views){
    view.matrix=new THREE.Matrix4().set(...view.pose.flat());
    view.position=new THREE.Vector3().setFromMatrixPosition(view.matrix);
    view.forward=new THREE.Vector3(0,0,-1).transformDirection(view.matrix);
  }
  model={materials:[]};
  async function loadScan() {
    const gltf=await new GLTFLoader().loadAsync('./'+modelFile+'?v=railrepair1');
    gltf.scene.traverse(obj=>{
      if(!obj.isMesh)return;
      const mat=obj.material;
      model.materials.push(mat);photoProjection(mat);
      if(mat.map)mat.map.anisotropy=Math.min(4,renderer.capabilities.getMaxAnisotropy());
    });
    scene.add(gltf.scene);
  }
  async function activate(kind,show) {
    const token=++action;
    const loading=$('loading');
    loading.hidden=false;loading.textContent=kind==='game'?'正在載入遊戲石階…':'正在載入實測場景…';
    try {
      if(kind==='game'){
        gameLoad??=createChurchGame(renderer).then(s=>{gameScene=s;}).catch(e=>{gameLoad=null;throw e;});
        await gameLoad;
      }else{
        scanLoad??=loadScan().catch(e=>{scanLoad=null;throw e;});
        await scanLoad;
      }
      if(token!==action)return;
      loading.hidden=true;show();dirty=true;ready=true;
    } catch(error) {
      if(token!==action)return;
      console.error(error);loading.textContent='場景載入失敗，請再按一次視圖按鈕重試。';
    }
  }
  function enterGame() {
    activate('game',()=>{
      if(!strolling()||!inChurchGame(camera.position.x,camera.position.z))positionAt(views.find(v=>v.id===198));
      setMode('game');renderer.shadowMap.needsUpdate=true;
      $('caption').textContent='遊戲重繪 · 石階';canvas.focus({preventScroll:true});
    });
  }
  for(const view of views.filter(v=>v.name)){
    const button=document.createElement('button');
    button.textContent=view.name;button.dataset.id=view.id;button.onclick=()=>activate('scan',()=>station(view));
    $('stations').append(button);
  }
  function openPhotoStop(index){
    activate('scan',()=>{
      routeOpen=true;stopIndex=THREE.MathUtils.clamp(index,0,photoStops.length-1);
      const view=views.find(v=>v.id===photoStops[stopIndex]);
      $('detail').checked=true;setMode('walk');positionAt(view,true);
      $('caption').textContent='實拍位置 '+(stopIndex+1)+' / '+photoStops.length;
      $('prev-photo').disabled=stopIndex===0;$('next-photo').disabled=stopIndex===photoStops.length-1;
      $('photo-progress').textContent=(stopIndex+1)+' / '+photoStops.length;
      $('help').textContent='前後拍攝點可看實拍細節。用方向鍵離開拍攝點，可看合成視角。';
      selectDetail();canvas.focus({preventScroll:true});
    });
  }
  function railSideView(){
    activate('scan',()=>{
      routeOpen=false;$('detail').checked=false;setMode('walk');
      positionAt(views.find(v=>v.id===199));
      camera.lookAt(2.28,-3.19,-16.55);
      turn.setFromQuaternion(camera.quaternion,'YXZ');highlight(null);railButton.setAttribute('aria-pressed','true');
      $('caption').textContent='欄杆側視';
      $('help').textContent='目前顯示模型內建貼圖，可左右移動檢查欄孔，也可勾選近景實拍。';
      selectDetail();canvas.focus({preventScroll:true});
    });
  }
  const railButton=document.createElement('button');
  railButton.textContent='欄杆側視';railButton.onclick=railSideView;$('stations').append(railButton);
  $('photo-route').onclick=()=>openPhotoStop(0);
  $('prev-photo').onclick=()=>openPhotoStop(stopIndex-1);
  $('next-photo').onclick=()=>openPhotoStop(stopIndex+1);
  $('game').onclick=enterGame;
  $('walk').onclick=()=>activate('scan',()=>{
    if(mode==='game'){
      setMode('walk');$('caption').textContent='室內漫遊';selectDetail();canvas.focus({preventScroll:true});
    }else station(current);
  });
  $('overview').onclick=()=>activate('scan',overview);
  $('compare').onclick=()=>{
    if(!ready)return;
    ++action;$('loading').hidden=true;
    if(mode==='compare'){
      if(photoReturn==='game'){
        setMode('game');$('caption').textContent='遊戲重繪 · 石階';
      }else if(photoReturn==='walk'&&routeOpen){
        setMode('walk');$('caption').textContent='實拍路線';selectDetail();
      }else station(current);
      return;
    }
    photoReturn=mode;
    if(strolling()){
      const direction=camera.getWorldDirection(tmp);
      const score=v=>v.position.distanceToSquared(camera.position)+4.2*(1-direction.dot(v.forward));
      current=views.reduce((a,b)=>score(b)<score(a)?b:a);
    }
    setMode('compare');highlight(current);
    $('reference').src='photo-'+current.id+'.webp';
    $('caption').textContent=(current.name||'鄰近拍攝點')+' · 原照';
  };
  $('detail').onchange=()=>{selectDetail();dirty=true;};
  $('bench').onclick=()=>{benchmark={start:performance.now(),last:0,frames:[]};$('bench').disabled=true;$('help').textContent='正在轉動視角量測幀率…';};
  canvas.addEventListener('pointerdown',e=>{
    if(!strolling())return;
    drag={id:e.pointerId,x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove',e=>{
    if(!drag||drag.id!==e.pointerId||!strolling())return;
    turn.y-=(e.clientX-drag.x)*.004;turn.x=THREE.MathUtils.clamp(turn.x-(e.clientY-drag.y)*.004,-1.42,1.42);
    camera.quaternion.setFromEuler(turn);drag.x=e.clientX;drag.y=e.clientY;dirty=true;
  });
  canvas.addEventListener('lostpointercapture',()=>{drag=null;});
  window.addEventListener('keydown',e=>{
    const key=moveKeys[e.code]||e.code;
    if(strolling()&&['KeyW','KeyA','KeyS','KeyD'].includes(key)&&!e.target.matches('input:not([type=checkbox]),textarea,select')){keys.add(key);e.preventDefault();}
  });
  window.addEventListener('keyup',e=>keys.delete(moveKeys[e.code]||e.code));
  window.addEventListener('blur',()=>keys.clear());
  for(const button of $('pad').children){
    button.onpointerdown=e=>{button.setPointerCapture(e.pointerId);keys.add(button.dataset.key);};
    button.onlostpointercapture=()=>keys.delete(button.dataset.key);
  }
  current=views.find(v=>v.id===(location.hash==='#photo-stairs'||location.hash==='#stairs-up'||location.hash==='#game-stairs'?198:21));
  if(location.hash==='#rail-repair')railSideView();
  else if(location.hash==='#photo-stairs')openPhotoStop(0);
  else if(location.hash==='#game-stairs')enterGame();
  else activate('scan',()=>station(current));
  requestAnimationFrame(step);
}
start().catch(error=>{console.error(error);$('loading').textContent='模型未能載入。可重新整理，或使用下方連結下載 GLB。';});
