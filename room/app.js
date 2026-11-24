import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomWalk } from './walk.js';
import { WalkPhotos } from './walk-photos.js';
import { loadSplats, loadSplatLayers, localCamera } from './local-splats.js';

const el=id=>document.getElementById(id);
const canvas=el('view');
const V=(x,y,z)=>new THREE.Vector3(x,y,z);
const home=V(.48,.64,-.12), orbit=V(5,5.6,-7.2);
const modes=['overview','plan','walk','inside'];
const names={overview:'全室剖視',plan:'平面俯視',walk:'全室漫遊',inside:'室內近看'};
const labels={full:'全室實拍模型',layout:'格局模型',scan:'原始掃描網格',splat:'實景高斯視差',live:'逐點高斯潑濺'};
const descriptions={
  full:'同一個完整三維模型。拖曳環繞、滾輪縮放，或進入全室漫遊。',
  walk:'拖曳轉頭，使用 WASD、方向鍵或畫面箭頭移動。',
  layout:'依掃描位置重畫床、書桌與家具。拖曳環繞，滾輪或雙指縮放。',
  scan:'由彩色影像與深度掃描重建。格局與原始細節可在「顯示方式」切換比較。',
  splat:'拖曳或按方向鍵可左右、上下小幅移動，觀察前後景視差。雙擊回到拍攝中心。'
};
const bound=n=>THREE.MathUtils.clamp(n,-1,1);

async function run(){
  const renderer=new THREE.WebGLRenderer({canvas,antialias:true,powerPreference:'high-performance'});
  renderer.setPixelRatio(Math.min(devicePixelRatio||1,1.5));
  renderer.localClippingEnabled=true;
  renderer.shadowMap.enabled=true;
  renderer.shadowMap.type=THREE.PCFShadowMap;
  renderer.shadowMap.autoUpdate=false;
  const scene=new THREE.Scene();scene.background=new THREE.Color('#e8e9e1');
  const splatScene=new THREE.Scene();splatScene.background=new THREE.Color('#242522');
  const photoCamera=new THREE.PerspectiveCamera(),camera=new THREE.PerspectiveCamera(42,1,.012,65);
  const controls=new OrbitControls(camera,canvas);
  controls.enableDamping=true;controls.enablePan=false;
  scene.add(new THREE.HemisphereLight(0xfffaf0,0xb6aba0,1.48));
  const sun=new THREE.DirectionalLight(0xfff3e2,2.1);sun.position.set(3,7,-4);scene.add(sun);
  sun.castShadow=true;sun.shadow.mapSize.set(1024,1024);
  Object.assign(sun.shadow.camera,{left:-4.5,right:4.5,top:4.5,bottom:-4.5,near:.2,far:16});
  sun.shadow.camera.updateProjectionMatrix();sun.shadow.normalBias=.013;sun.shadow.bias=-.0003;
  const fill=new THREE.DirectionalLight(0xd8eaff,1);fill.position.set(-4,4,3);scene.add(fill);
  const grid=new THREE.GridHelper(30,60,0xbcc3b7,0xcdd2c8);grid.position.y=-.177;
  grid.material.transparent=true;grid.material.opacity=.32;scene.add(grid);
  const roof=new THREE.Plane(V(0,-1,0),2.12);
  const bounds=[new THREE.Plane(V(1,0,0),2.38),new THREE.Plane(V(-1,0,0),3.42),
    new THREE.Plane(V(0,0,1),2.48),new THREE.Plane(V(0,0,-1),2.36),new THREE.Plane(V(0,1,0),.09)];
  const response=await fetch('./scene.json');if(!response.ok)throw Error('場景說明下載失敗');
  const data=await response.json(),cache=new Map(),loader=new GLTFLoader();
  const stationButtons=[];
  let mode='overview',station=0,overviewMethod='full',insideMethod='splat';
  let active=null,dirty=true,comparing=false,bench=null,generation=0,dx=0,dy=0,pointer=null;
  let loaded=false,lastWidth=0,lastHeight=0;
  const method=()=>mode==='walk'?'full':mode==='inside'?insideMethod:overviewMethod;
  const gridResponse=await fetch('./walk-grid.json');
  if(!gridResponse.ok)throw Error('無法載入房間動線');
  const walk=new RoomWalk(camera,await gridResponse.json(),canvas,el('walk-pad'),()=>{dirty=true;});
  const walkPhotos=new WalkPhotos();
  const isLocal=()=>['splat','live'].includes(method());
  controls.addEventListener('change',()=>{dirty=true;});

  function load(key){
    if(!cache.has(key)){
      const promise=key.startsWith('splat-')?loadSplatLayers(Number(key.slice(6))):
        key.startsWith('live-')?loadSplats(Number(key.slice(5))):
        loader.loadAsync(key==='full'?'./bedroom-full.glb?v=photo2':key==='layout'?'./bedroom-layout.glb':'./bedroom.glb').then(a=>a.scene);
      cache.set(key,promise);
      promise.catch(()=>cache.delete(key));
    }
    return cache.get(key);
  }
  function distance(){return orbit.length()*Math.max(1,1.40/Math.max(.5,camera.aspect));}
  function resetCamera(){
    if(mode==='walk'){controls.enabled=false;walk.enter();return;}
    controls.enableDamping=false;controls.update();controls.enableDamping=true;
    dx=dy=0;camera.fov=mode==='inside'?49.13:41;camera.updateProjectionMatrix();
    const interior=mode==='inside';
    controls.enabled=!isLocal()&&!bench;
    controls.enableRotate=mode!=='plan';controls.enableZoom=!interior;
    controls.minDistance=interior?.02:3;controls.maxDistance=interior?.06:35;
    controls.minPolarAngle=.001;controls.maxPolarAngle=interior?Math.PI-.001:Math.PI*.48;
    controls.rotateSpeed=interior?-.48:.75;
    if(interior){
      const entry=data.stations[station];controls.target.fromArray(entry.position);
      camera.position.copy(controls.target).addScaledVector(new THREE.Vector3().fromArray(entry.direction),-.038);
    }else{
      controls.target.copy(home);
      const offset=mode==='plan'?V(0,1,-.001):orbit.clone().normalize();
      camera.position.copy(home).addScaledVector(offset,distance()*(mode==='plan'?.90:1));
    }
    controls.update();dirty=true;
  }
  function showPhoto(show){
    comparing=show;
    const photo=el('reference');
    photo.src=data.stations[station].photo;photo.hidden=!show;
    el('compare').textContent=show?'回到 3D 視角':'對照原始照片';
    el('compare').setAttribute('aria-pressed',String(show));
    dirty=true;
  }
  function updateLabels(){
    for(const m of modes)el(m).setAttribute('aria-pressed',String(m===mode));
    stationButtons.forEach((btn,i)=>btn.setAttribute('aria-pressed',String(mode==='inside'&&station===i)));
    el('method').replaceChildren();
    for(const opt of (mode==='walk'?['full']:mode==='inside'?['splat','live','scan']:['full','scan','layout'])){
      el('method').add(new Option(labels[opt],opt));
    }
    el('method').value=method();el('method').disabled=mode==='walk';el('compare').hidden=mode!=='inside';
    const layout=method()==='layout',full=method()==='full';
    el('download').href=full?'./bedroom-full.glb?v=photo2':layout?'./bedroom-layout.glb':'./bedroom.glb';
    el('download').textContent=full?'下載全室 GLB ↓':layout?'下載格局 GLB ↓':'下載掃描 GLB ↓';
    canvas.setAttribute('aria-label',mode==='walk'?'拖曳轉頭，WASD 或方向鍵移動':isLocal()?'拖曳或用方向鍵移動視角':'拖曳環繞整間臥室，滾輪或雙指縮放');
    el('note').textContent=descriptions[mode==='walk'?'walk':method()==='live'?'splat':method()];
  }
  async function select(next,index=station){
    const ticket=++generation;
    if(bench)finish('');
    mode=next;station=index;loaded=false;pointer=null;walk.setEnabled(false);
    showPhoto(false);updateLabels();resetCamera();
    el('state').textContent='載入中…';el('test').disabled=true;
    try{
      const kind=method(),key=isLocal()?kind+'-'+data.stations[station].id:kind;
      const object=await load(key);
      if(ticket!==generation)return;
      if(kind==='full'){
        if(mode==='walk')walkPhotos.load();
        object.traverse(mesh=>{
          if(!mesh.isMesh)return;
          mesh.userData.atlasMaterial??=mesh.material;
          if(mode==='walk')mesh.userData.walkMaterial??=walkPhotos.material(mesh.userData.atlasMaterial);
          mesh.material=mode==='walk'?mesh.userData.walkMaterial:mesh.userData.atlasMaterial;
        });
      }
      if(active)active.removeFromParent();
      active=object;
      if(isLocal())splatScene.add(object);
      else{
        scene.add(object);
        if(kind==='layout'){
          const cutHeight=new THREE.Plane(V(0,-1,0),2.58);
          object.traverse(o=>{
            if(!o.isMesh)return;
            let shell=false;
            for(let parent=o.parent;parent;parent=parent.parent)if(parent.name.startsWith('Shell_'))shell=true;
            o.receiveShadow=true;o.castShadow=!shell;
            o.material.clippingPlanes=[cutHeight];
          });
          renderer.shadowMap.needsUpdate=true;
        }
        if(kind==='scan'||kind==='full'){
          roof.constant=mode==='plan'?1.45:2.12;
          object.traverse(o=>{if(o.isMesh){o.material.clippingPlanes=mode==='walk'?[]:mode==='inside'?bounds:[...bounds,roof];o.material.needsUpdate=true;}});
        }
      }
      grid.visible=!['walk','inside'].includes(mode);walk.setEnabled(mode==='walk');loaded=true;dirty=true;el('test').disabled=false;
      if(mode==='walk')canvas.focus({preventScroll:true});
      el('state').textContent=isLocal()?'實景視差 · '+data.stations[station].name:names[mode]+' · '+labels[kind];
    }catch(err){
      if(ticket===generation){el('state').textContent='載入失敗';el('note').textContent=err.message+'。可重新選擇位置或顯示方式。';}
    }
  }
  data.stations.forEach((s,i)=>{
    const button=document.createElement('button');button.className='station';
    const image=new Image();image.src=s.photo;image.alt='';
    button.append(image,document.createTextNode(s.name));
    button.onclick=()=>navigate('inside',i);stationButtons.push(button);el('stations').append(button);
  });
  function navigate(next,index=station){
    const hash=next==='overview'?'':next==='inside'?links[index]:'#'+next;
    if(location.hash!==hash)history.pushState(null,'',location.pathname+location.search+hash);
    return select(next,index);
  }
  modes.forEach(m=>el(m).onclick=()=>navigate(m));
  el('method').onchange=e=>{
    if(mode==='inside')insideMethod=e.target.value;else overviewMethod=e.target.value;
    select(mode);
  };
  el('compare').onclick=()=>showPhoto(!comparing);
  const canMove=()=>loaded && isLocal() && !comparing && !bench;
  canvas.addEventListener('pointerdown',e=>{
    if(!e.isPrimary||!canMove())return;
    pointer={id:e.pointerId,x:e.clientX,y:e.clientY,dx,dy};
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove',e=>{
    if(!pointer||pointer.id!==e.pointerId||!canMove())return;
    dx=bound(pointer.dx+(e.clientX-pointer.x)/(canvas.clientWidth*.28));
    dy=bound(pointer.dy-(e.clientY-pointer.y)/(canvas.clientHeight*.28));
    dirty=true;
  });
  canvas.addEventListener('lostpointercapture',()=>{pointer=null;});
  canvas.ondblclick=()=>{if(canMove()){dx=dy=0;dirty=true;}};
  canvas.onkeydown=e=>{
    if(!canMove())return;
    const steps={ArrowLeft:[-.2,0],ArrowRight:[.2,0],ArrowDown:[0,-.2],ArrowUp:[0,.2],Home:[-dx,-dy]};
    if(!steps[e.key])return;
    e.preventDefault();dx=bound(dx+steps[e.key][0]);dy=bound(dy+steps[e.key][1]);dirty=true;
  };
  function size(){
    const w=canvas.clientWidth,h=canvas.clientHeight;
    if(w===lastWidth&&h===lastHeight)return;
    lastWidth=w;lastHeight=h;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();
    if(bench)finish('畫布大小已改變，請重新測試。');
    if(mode==='walk'){camera.fov=68;camera.updateProjectionMatrix();dirty=true;}else resetCamera();
  }
  function lock(value){
    [el('test'),el('method'),el('compare'),...modes.map(el),...stationButtons].forEach(b=>b.disabled=value);
    el('method').disabled=value||mode==='walk';
    walk.setEnabled(!value&&mode==='walk');
    controls.enabled=!value&&!isLocal()&&mode!=='walk';
  }
  function finish(message){bench=null;lock(false);resetCamera();el('note').textContent=message;el('fps').textContent='';}
  el('test').onclick=()=>{
    if(!loaded)return;
    showPhoto(false);bench={start:null,last:null,frames:[]};lock(true);
    el('note').textContent='暖機 2 秒，再量測目前視圖 10 秒。請保持分頁在前景。';
  };
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&bench)finish('測試已中止，請維持前景後重測。');});
  function animateTest(now){
    bench.start??=now;const ms=now-bench.start,phase=ms/12000*2*Math.PI;
    if(mode==='walk'){
      camera.rotation.set(-.13+.06*Math.sin(phase),walk.yaw+.68*Math.sin(phase),0,'YXZ');
    }else if(isLocal()){dx=Math.sin(phase);dy=.6*Math.cos(phase);}
    else if(mode==='inside'){
      const s=data.stations[station];const dir=new THREE.Vector3().fromArray(s.direction).applyAxisAngle(V(0,1,0),.3*Math.sin(phase));
      camera.position.copy(controls.target).addScaledVector(dir,-.038);camera.lookAt(controls.target);
    }else{
      const d=mode==='plan'?V(0,1,-.001).multiplyScalar(distance()*(.88+.06*Math.sin(phase))):
        orbit.clone().normalize().multiplyScalar(distance()).applyAxisAngle(V(0,1,0),phase);
      camera.position.copy(home).add(d);camera.lookAt(home);
    }
    dirty=true;
    if(ms>=2000){if(bench.last!==null)bench.frames.push(now-bench.last);bench.last=now;}
    if(ms>=12000&&bench.frames.length){
      const times=bench.frames.sort((a,b)=>a-b),fps=1000*times.length/times.reduce((a,b)=>a+b,0);
      finish(labels[method()]+' · '+canvas.width+' × '+canvas.height+' · 平均 '+fps.toFixed(1)+' FPS · P95 '+times[Math.ceil(times.length*.95)-1].toFixed(1)+' ms');
    }
  }
  size();new ResizeObserver(size).observe(canvas);
  const links=['#bed','#desk','#mirror','#rug'];
  function followLink(){
    const deep=links.indexOf(location.hash);
    return select(location.hash==='#walk'?'walk':location.hash==='#plan'?'plan':deep>=0?'inside':'overview',deep<0?0:deep);
  }
  window.addEventListener('hashchange',followLink);
  await followLink();
  renderer.setAnimationLoop(now=>{
    if(!loaded)return;
    if(bench)animateTest(now);else if(mode==='walk')walk.tick(now);else if(controls.enabled)controls.update();
    if(mode==='walk'&&walkPhotos.update(camera,now))dirty=true;
    if(!dirty||comparing)return;
    const kind=method(),x=camera.position.x-home.x,z=camera.position.z-home.z;
    if(isLocal()){
      localCamera(photoCamera,camera.aspect,dx,dy);
      if(kind==='live')active.material.uniforms.rasterHeight.value=canvas.height;
      renderer.render(splatScene,photoCamera);
    }else{
      if(kind==='layout'){
        const plan=mode==='plan';
        for(const [name,visible] of Object.entries({ceiling:false,back:!plan&&z<1.1,front:!plan&&z>-1.1,left:!plan&&x>-1.1,right:!plan&&x<1.1})){
          const obj=active.getObjectByName('Shell_'+name);if(obj)obj.visible=visible;
        }
      }else{
        const cut=mode==='overview';
        bounds[0].constant=cut&&x< -1.15?1.94:2.38;bounds[1].constant=cut&&x>1.15?2.96:3.42;
        bounds[2].constant=cut&&z< -1.15?2.03:2.48;bounds[3].constant=cut&&z>1.15?1.91:2.36;
      }
      renderer.render(scene,camera);
    }
    dirty=false;
  });
}
run().catch(error=>{el('state').textContent='無法開啟 3D 場景';el('note').textContent=error.message;});
