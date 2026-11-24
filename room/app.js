import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const byId = id => document.getElementById(id);
const canvas = byId('view');
const V = (x,y,z) => new THREE.Vector3(x,y,z);
const home = V(.48,.6,-.12);
const overviewOffset = V(5,5.4,-7.2);
const modes = ['overview','plan','inside'];

async function run() {
  const renderer = new THREE.WebGLRenderer({canvas,antialias:false,powerPreference:'high-performance'});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1,1.25));
  renderer.localClippingEnabled = true;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#e8e9e1');
  const camera = new THREE.PerspectiveCamera(42,1,.012,65);
  const controls = new OrbitControls(camera,canvas);
  controls.enableDamping=true;
  controls.enablePan=false;
  const response=await fetch('./scene.json');
  if(!response.ok) throw Error('場景資料載入失敗');
  const data=await response.json();
  const asset=await new GLTFLoader().loadAsync('./bedroom.glb');
  scene.add(asset.scene);

  // The downloadable scene includes a separately named inferred envelope.
  const base=new THREE.Mesh(new THREE.BoxGeometry(5.84,.075,4.85),new THREE.MeshBasicMaterial({color:0xb0b1a2}));
  base.position.set(.53,-.135,-.075);
  scene.add(base);
  const grid=new THREE.GridHelper(30,60,0xbcc3b7,0xcdd2c8);
  grid.position.y=-.177;
  grid.material.transparent=true;grid.material.opacity=.32;
  scene.add(grid);
  const materials=[];
  asset.scene.traverse(o=>{if(o.isMesh)materials.push(o.material);});
  const clipping=new THREE.Plane(V(0,-1,0),2.0);
  // Exclude isolated scanning artifacts outside the observed room footprint.
  const bounds=[
    new THREE.Plane(V(1,0,0),2.38),new THREE.Plane(V(-1,0,0),3.42),
    new THREE.Plane(V(0,0,1),2.48),new THREE.Plane(V(0,0,-1),2.36),
    new THREE.Plane(V(0,1,0),.09)
  ];
  let dirty=true;
  controls.addEventListener('change',()=>{dirty=true;});
  let mode='overview', station=0, comparing=false, benchmark=null;
  const stationButtons=[];
  const directions={
    overview:'拖曳可環繞整間臥室，滾輪或雙指可縮放。',
    plan:'從上方檢視床、書桌與座椅的位置。切換拍攝位置可進入室內。',
    inside:'拖曳環視，點「對照原始照片」可查看同一位置的實拍影像。'
  };
  const resetPhoto = () => {
    comparing=false;
    byId('reference').hidden=true;
    byId('compare').textContent='對照原始照片';
    byId('compare').setAttribute('aria-pressed','false');
  };
  function fitDistance() {
    const horizontalFit=Math.max(1,1.42/Math.max(.5,camera.aspect));
    return overviewOffset.length()*horizontalFit;
  }
  function setView(next, index=station) {
    dirty=true;mode=next; station=index;
    controls.enableDamping=false;
    controls.update();
    resetPhoto();
    for(const key of modes) byId(key).setAttribute('aria-pressed',String(key===mode));
    stationButtons.forEach((button,i)=>button.setAttribute('aria-pressed',String(mode==='inside'&&i===station)));
    byId('compare').hidden = mode!=='inside';
    grid.visible=base.visible=mode!=='inside';
    clipping.constant=mode==='plan'?1.45:2.12;
    for(const mat of materials){
      mat.clippingPlanes=mode==='inside'?bounds:[...bounds,clipping];
      mat.needsUpdate=true;
    }
    camera.fov=mode==='inside'?49.13:42;
    camera.updateProjectionMatrix();
    controls.enableRotate=mode!=='plan';
    controls.enableZoom=mode!=='inside';
    controls.minPolarAngle=.001;
    controls.maxPolarAngle=mode==='inside'?Math.PI-.001:Math.PI*.48;
    controls.minDistance=mode==='inside'?.02:3.0;
    controls.maxDistance=mode==='inside'?.06:32;
    controls.rotateSpeed=mode==='inside'?-.48:.75;
    if(mode==='inside'){
      const capture=data.stations[station];
      controls.target.fromArray(capture.position);
      camera.position.copy(controls.target).addScaledVector(new THREE.Vector3().fromArray(capture.direction),-.038);
      byId('state').textContent='室內視角 · '+capture.name;
    }else{
      controls.target.copy(home);
      const offset=mode==='plan'?V(0,1,-.001):overviewOffset.clone().normalize();
      camera.position.copy(home).addScaledVector(offset,fitDistance()*(mode==='plan'?.89:1));
      byId('state').textContent=mode==='plan'?'平面俯視':'全室剖視';
    }
    controls.update();
    controls.enableDamping=true;
    byId('note').textContent=directions[mode];
  }

  data.stations.forEach((entry,i)=>{
    const btn=document.createElement('button');btn.className='station';
    const img=new Image();img.src=entry.photo;img.alt='';
    btn.append(img,document.createTextNode(entry.name));
    btn.onclick=()=>setView('inside',i);
    stationButtons.push(btn);byId('stations').append(btn);
  });
  modes.forEach(key=>byId(key).onclick=()=>setView(key));
  byId('compare').onclick=()=>{
    const show=!comparing;
    setView('inside');
    if(show){
      comparing=true;
      const img=byId('reference');img.src=data.stations[station].photo;img.hidden=false;
      byId('compare').textContent='回到 3D 視角';
      byId('compare').setAttribute('aria-pressed','true');
      byId('state').textContent='拍攝原圖 · '+data.stations[station].name;
    }
  };

  function lock(value) {
    [byId('test'),byId('compare'),...modes.map(byId),...stationButtons].forEach(btn=>btn.disabled=value);
    controls.enabled=!value;
  }
  byId('test').onclick=()=>{
    setView('overview');lock(true);
    benchmark={start:null,last:null,samples:[]};
    byId('note').textContent='正在環繞全室。暖機 2 秒，再量測 10 秒，請維持前景顯示。';
  };
  function endBenchmark(message) {
    benchmark=null;lock(false);setView('overview');byId('note').textContent=message;
  }
  document.addEventListener('visibilitychange',()=>{
    if(document.hidden && benchmark)endBenchmark('測試已暫停。請在前景分頁重新測試。');
  });
  function size(){
    const w=canvas.clientWidth,h=canvas.clientHeight;
    renderer.setSize(w,h,false);
    camera.aspect=w/h;camera.updateProjectionMatrix();
  }
  size();setView('overview');
  new ResizeObserver(()=>{
    if(benchmark)endBenchmark('畫布大小已變更，請重新測試。');
    size();dirty=true;
    if(mode!=='inside')setView(mode);
  }).observe(canvas);
  let since=null,count=0;
  renderer.setAnimationLoop(now=>{
    if(benchmark){
      dirty=true;
      benchmark.start??=now;
      const elapsed=now-benchmark.start;
      const offset=overviewOffset.clone().normalize().multiplyScalar(fitDistance());
      offset.applyAxisAngle(V(0,1,0),elapsed/12000*Math.PI*2);
      camera.position.copy(home).add(offset);camera.lookAt(home);
      if(elapsed>=2000){
        if(benchmark.last!==null)benchmark.samples.push(now-benchmark.last);
        benchmark.last=now;
      }
      if(elapsed>=12000 && benchmark.samples.length){
        const times=benchmark.samples.sort((a,b)=>a-b),fps=1000*times.length/times.reduce((s,n)=>s+n,0);
        endBenchmark(canvas.width+' × '+canvas.height+' px · 平均 '+fps.toFixed(1)+' FPS · P95 '+times[Math.ceil(times.length*.95)-1].toFixed(1)+' ms');
      }
    }else controls.update();
    // Peel the two near walls in the dollhouse view. Keep the full outer
    // wall and ceiling available when looking out from a capture station.
    const relativeX=camera.position.x-home.x, relativeZ=camera.position.z-home.z;
    const orbit=mode==='overview';
    bounds[0].constant=orbit && relativeX < -1.15 ? 1.94:2.38;
    bounds[1].constant=orbit && relativeX >  1.15 ? 2.96:3.42;
    bounds[2].constant=orbit && relativeZ < -1.15 ? 2.03:2.48;
    bounds[3].constant=orbit && relativeZ >  1.15 ? 1.91:2.36;
    if(dirty && !comparing){renderer.render(scene,camera);dirty=false;}
    since??=now;
    if(now-since>=1000){byId('fps').textContent=benchmark?Math.round(1000*count/(now-since))+' FPS':'';since=now;count=0;}
    count++;
  });
}
run().catch(err=>{
  byId('state').textContent='無法載入 3D 場景';
  byId('note').textContent=err.message+'。可下載 GLB 模型至 Blender 檢視。';
});
