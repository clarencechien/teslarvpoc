import * as THREE from 'three';
import {scanProjection} from './church-scan-projection.js';
import {photoSurface} from './church-photo-surface.js';

// Unproject a regular grid through the calibrated equidistant fisheye lens. The
// lookup is shared by all photographs from a camera, while range changes per exposure.
const raysByLens=new Map();
function gridRays(view,width,height) {
  const key=[width,height,...view.lens,...view.distortion].join(',');
  if(raysByLens.has(key)) return raysByLens.get(key);
  const [fx,fy,cx,cy]=view.lens,[k1,k2,k3,k4]=view.distortion;
  const rays=new Float32Array(width*height*3);
  for(let i=0;i<width*height;i++){
    const x=((((i%width)+.5)*1440/width-.5)-cx)/fx;
    const y=(((Math.floor(i/width)+.5)*1080/height-.5)-cy)/fy;
    const radius=Math.hypot(x,y);
    let angle=radius;
    for(let j=0;j<7;j++){
      const s=angle*angle;
      const value=angle*(1+s*(k1+s*(k2+s*(k3+s*k4))))-radius;
      angle-=value/(1+s*(3*k1+s*(5*k2+s*(7*k3+9*s*k4))));
    }
    const radial=Math.sin(angle)/Math.max(radius,1e-8);
    rays.set([x*radial,-y*radial,-Math.cos(angle)],3*i);
  }
  raysByLens.set(key,rays);return rays;
}

const WANTED = 4;
const LAYERS = WANTED + 1; // A spare layer lets a source enter before the previous one leaves.
const ids = Array.from({length: LAYERS}, (_, i) => i);

function makeLayerMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: {photo: {value: null}, sourceCenter: {value: new THREE.Vector3()}},
    vertexShader: `
      attribute float certainty;
      varying vec2 st;
      varying vec3 world;
      varying float support;
      void main() {
        st = uv; support = certainty;
        world = (modelMatrix * vec4(position, 1.)).xyz;
        gl_Position = projectionMatrix * viewMatrix * vec4(world, 1.);
      }`,
    fragmentShader: `
      uniform sampler2D photo;
      uniform vec3 sourceCenter;
      varying vec2 st;
      varying vec3 world;
      varying float support;
      void main() {
        float border = min(min(st.x, 1.-st.x), min(st.y, 1.-st.y));
        vec3 sourceRay = normalize(world-sourceCenter), targetRay = normalize(world-cameraPosition);
        float parallax = length(sourceRay-targetRay);
        float reliability = smoothstep(.035,.64,support) * smoothstep(.002,.049,border)
                          * (1.-smoothstep(.23,.75,parallax));
        // Reject elongated projected texels where a foreground silhouette bridges a background gap.
        vec2 sourcePixel = st * vec2(1440.,1080.);
        vec2 dx=dFdx(sourcePixel), dy=dFdy(sourcePixel);
        float trace=dot(dx,dx)+dot(dy,dy), area=abs(dx.x*dy.y-dx.y*dy.x);
        float anisotropy=(trace+sqrt(max(0.,trace*trace-4.*area*area)))/max(.00001,2.*area);
        float drop=cameraPosition.y-world.y;
        float furniture=smoothstep(3.,6.,world.z)*smoothstep(.2,.65,drop)*(1.-smoothstep(1.9,2.6,drop))
                       *(1.-smoothstep(6.,11.,distance(cameraPosition,world)));
        reliability *= 1.-smoothstep(2.2,5.0,anisotropy)*(1.-.55*furniture);
        if(reliability < .008) discard;
        gl_FragColor = vec4(texture2D(photo,st).rgb, reliability);
        #include <colorspace_fragment>
      }`
  });
}

function compositor() {
  const uniforms = {worldFromClip:{value:new THREE.Matrix4()}, scanDepth:{value:null}, scanColor:{value:null}, clip: {value: new THREE.Vector2()}, weights: {value: new Float32Array(LAYERS)},
    fades: {value: new Float32Array(LAYERS)}, ramp:{value:new Float32Array(LAYERS)}};
  for (const i of ids) {
    uniforms['layer'+i] = {value: null};
    uniforms['depth'+i] = {value: null};
  }
  return new THREE.ShaderMaterial({
    uniforms, transparent: true, depthTest: false, depthWrite: false,
    vertexShader: 'varying vec2 st;void main(){st=position.xy*.5+.5;gl_Position=vec4(position.xy,0.,1.);}',
    fragmentShader: `
      varying vec2 st;uniform vec2 clip;
      uniform mat4 worldFromClip;uniform sampler2D scanDepth,scanColor;uniform float weights[${LAYERS}], fades[${LAYERS}],ramp[${LAYERS}];
      ${ids.map(i=>'uniform sampler2D layer'+i+',depth'+i+';').join('\n')}
      float metres(float z){return clip.x*clip.y/(clip.y-z*(clip.y-clip.x));}
      void main(){
        vec4 pixels[${LAYERS}];float d[${LAYERS}];
        ${ids.map(i=>`pixels[${i}]=texture2D(layer${i},st);pixels[${i}].a*=fades[${i}];
                        d[${i}]=metres(texture2D(depth${i},st).r);`).join('\n')}
        float raw=texture2D(scanDepth,st).r, measured=metres(raw);
        vec4 p=worldFromClip*vec4(st*2.-1.,raw*2.-1.,1.);
        vec3 world=p.xyz/p.w;
        float drop=cameraPosition.y-world.y;
        float furniture=smoothstep(3.,6.,world.z)*smoothstep(.2,.65,drop)*(1.-smoothstep(1.9,2.6,drop))
                       *(1.-smoothstep(6.,11.,distance(cameraPosition,world)));
        vec3 sum=vec3(0.);float total=0., coverage=0., rawWeight[${LAYERS}];
        for(int i=0;i<${LAYERS};i++){
          float delta=abs(d[i]-measured);
          float agreement=1.-smoothstep(.25+.045*measured,.70+.13*measured,delta);
          float visible=pixels[i].a * (measured>clip.y*.98 ? 1. : mix(.12+.82*furniture,1.,agreement));
          for(int j=0;j<${LAYERS};j++){
            // A fading or marginal observation must not suddenly occlude a wall.
            float front=smoothstep(.09+.015*d[i],.35+.04*d[i],d[i]-d[j]);
            float authority=weights[j]/max(.001,weights[j]+.30*weights[i]);
            visible*=1.-front*pixels[j].a*authority;
          }
          float w=weights[i]*visible;
          rawWeight[i]=w;total+=w;coverage+=visible;
        }
        float weighted=0.;
        for(int i=0;i<${LAYERS};i++){
          float other=max(0.,total-rawWeight[i]);
          float w=rawWeight[i];
          if(ramp[i]<.999 && other>.0001)w=min(w,other*ramp[i]/max(.001,1.-ramp[i]));
          sum+=w*pixels[i].rgb;weighted+=w;
        }
        vec4 base=texture2D(scanColor,st);
        // Give a well-supported, sharp scan surface precedence at grazing angles
        // and distant masonry; keep nearby photo geometry through missing furniture.
        float stable=base.a*(1.-.95*furniture)*mix(.22,.90,smoothstep(2.5,8.,measured));
        gl_FragColor=vec4(mix(base.rgb,sum/max(.00001,weighted),(1.-stable)*smoothstep(.015,.93,coverage)),1.);
        #include <colorspace_fragment>
      }`
  });
}

function makeTarget() {
  const target = new THREE.WebGLRenderTarget(1, 1, {
    minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter,
    depthTexture: new THREE.DepthTexture(1, 1, THREE.UnsignedIntType)
  });
  target.texture.colorSpace = THREE.SRGBColorSpace;
  return target;
}

export class PhotoWalk {
  constructor(views,invalidate) {
    this.views=views; this.invalidate=invalidate;
    this.cache=new Map(); this.wanted=[]; this.retryAt=0; this.loading=false; this.enabled=false;
    this.layers=ids.map(()=>({item:null, fade:0, target:makeTarget()}));
    this.scanTarget=makeTarget();
    this.projected=new THREE.WebGLRenderTarget(1,1,{depthBuffer:false});
    this.projected.texture.colorSpace=THREE.SRGBColorSpace;
    this.onScan=scanProjection(ids);
    this.layerScene=new THREE.Scene(); this.surfaceMaterial=makeLayerMaterial();
    this.surface=new THREE.Mesh(undefined,this.surfaceMaterial);
    this.surface.matrixAutoUpdate=false; this.surface.frustumCulled=false; this.layerScene.add(this.surface);
    this.screen=new THREE.Scene(); this.blend=compositor();
    const quad=this.quad=new THREE.Mesh(new THREE.PlaneGeometry(2,2),this.blend);
    quad.frustumCulled=false; this.screen.add(quad);
    this.loader=new THREE.TextureLoader(); this.direction=new THREE.Vector3();
    this.size=new THREE.Vector2(); this.clearColor=new THREE.Color(); this.lastFrame=0;
  }
  get ready() {return this.layers.some(layer=>layer.item!==null);}
  resetStation() {
    this.wanted=[];
    for(const layer of this.layers){layer.item=null;layer.fade=0;}
  }
  static async create(invalidate) {
    const response=await fetch('./walk-light/views.json');
    if(!response.ok)throw Error('實拍索引讀取失敗');
    const sources=await response.json();
    for(const view of sources){
      view.matrix=new THREE.Matrix4().set(...view.pose.flat());
      view.inverse=view.matrix.clone().invert();
      view.position=new THREE.Vector3().setFromMatrixPosition(view.matrix);
      view.forward=new THREE.Vector3(0,0,-1).transformDirection(view.matrix);
    }
    return new PhotoWalk(sources,invalidate);
  }
  async load(view) {
    const revision=view.id===199?'?v=landing2':'';
    const [photo,range]=await Promise.allSettled(['webp','png'].map(ext=>
      this.loader.loadAsync('./walk-light/'+view.id+'.'+ext+revision)));
    if(photo.status==='rejected'||range.status==='rejected'){
      if(photo.status==='fulfilled')photo.value.dispose();
      if(range.status==='fulfilled')range.value.dispose();
      throw Error('實拍圖層讀取失敗');
    }
    const texture=photo.value, image=range.value.image;
    try {
      const canvas=document.createElement('canvas');
      const w=canvas.width=image.width, h=canvas.height=image.height;
      const ctx=canvas.getContext('2d',{willReadFrequently:true});
      ctx.drawImage(image,0,0);
      const geometry=photoSurface(ctx.getImageData(0,0,w,h).data,gridRays(view,w,h),w,h);
      texture.flipY=false; texture.colorSpace=THREE.SRGBColorSpace;
      const depth=range.value;
      depth.flipY=false;depth.generateMipmaps=false;depth.minFilter=depth.magFilter=THREE.NearestFilter;
      this.cache.set(view.id,{view,geometry,texture,depth});
    } catch(error) {texture.dispose();range.value.dispose();throw error;}
  }
  select(camera) {
    if(!this.enabled)return;
    const forward=camera.getWorldDirection(this.direction);
    const scored=this.views.map(view=>({view,score:camera.position.distanceToSquared(view.position)+
      4.2*(1-forward.dot(view.forward))-(this.wanted.includes(view)?.14:0)}));
    scored.sort((a,b)=>a.score-b.score);
    this.wanted=scored.slice(0,WANTED).map(entry=>entry.view);
    this.prepare();
  }
  prepare() {
    if(!this.enabled)return;
    const missing=this.wanted.filter(view=>!this.cache.has(view.id));
    if(missing.length && !this.loading && performance.now()>=this.retryAt){
      this.loading=true;
      Promise.allSettled(missing.map(view=>this.load(view))).then(results=>{
        this.loading=false;
        if(results.some(r=>r.status==='rejected'))this.retryAt=performance.now()+1800;
        this.prepare();this.trim();this.invalidate();
      });
    }
    // Commit against the current camera selection, not the selection that started a download.
    if(!missing.length || (!this.ready && performance.now()<this.retryAt)){
      const first=!this.ready;
      for(const view of this.wanted){
        const item=this.cache.get(view.id);
        if(!item || this.layers.some(layer=>layer.item===item))continue;
        const spare=this.layers.find(layer=>!layer.item);
        if(!spare)break;
        spare.item=item;spare.fade=first?1:0;
        this.invalidate();
      }
    }
  }
  trim() {
    for(const [id,item] of this.cache){
      if(this.cache.size<=10)break;
      if(this.wanted.includes(item.view)||this.layers.some(layer=>layer.item===item))continue;
      item.geometry.dispose();item.texture.dispose();item.depth.dispose();this.cache.delete(id);
    }
  }
  render(renderer,camera,scan) {
    const now=performance.now();
    const step=Math.min(.042,Math.max(0,(now-this.lastFrame)/1000))/.40;
    this.lastFrame=now;
    const canRetire=this.wanted.every(view=>this.cache.has(view.id));
    const waitingForSlot=this.wanted.some(view=>!this.layers.some(layer=>layer.item?.view===view));
    const outgoing=this.layers.filter(layer=>layer.item&&!this.wanted.includes(layer.item.view))
      .sort((a,b)=>a.fade-b.fade || camera.position.distanceToSquared(b.item.view.position)-camera.position.distanceToSquared(a.item.view.position));
    let changing=false;
    for(const layer of this.layers){
      if(!layer.item)continue;
      const wanted=this.wanted.includes(layer.item.view);
      const retire=canRetire && (!waitingForSlot || layer===outgoing[0]);
      const goal=wanted?1:retire?0:layer.fade;
      if(layer.fade!==goal){
        layer.fade=THREE.MathUtils.clamp(goal>layer.fade?layer.fade+step:layer.fade-step,0,1);
        if(!layer.fade&&!goal)layer.item=null;
        changing=true;
      }
    }
    if(changing){this.prepare();this.trim();this.invalidate();}
    renderer.getDrawingBufferSize(this.size);
    const scale=Math.min(1,Math.sqrt(1800000/(this.size.x*this.size.y)));
    const w=Math.max(1,Math.round(this.size.x*scale)),h=Math.max(1,Math.round(this.size.y*scale));
    this.scanTarget.setSize(this.size.x,this.size.y);
    renderer.setRenderTarget(this.scanTarget);renderer.render(scan,camera);
    this.blend.uniforms.scanDepth.value=this.scanTarget.depthTexture;
    const scanU=this.onScan.uniforms;
    scanU.zMap.value=this.scanTarget.depthTexture;scanU.base.value=this.scanTarget.texture;
    scanU.eye.value.multiplyMatrices(camera.matrixWorld,camera.projectionMatrixInverse);
    this.blend.uniforms.worldFromClip.value.copy(scanU.eye.value);
    const oldAlpha=renderer.getClearAlpha();renderer.getClearColor(this.clearColor);
    renderer.setClearColor(0,0);
    const u=this.blend.uniforms;
    camera.getWorldDirection(this.direction);u.clip.value.set(camera.near,camera.far);
    for(const i of ids){
      const layer=this.layers[i],item=layer.item;
      u.weights.value[i]=0;u.fades.value[i]=0;scanU.fades.value[i]=0;
      if(!item || layer.fade<=0)continue;
      const v=item.view,distance=camera.position.distanceTo(v.position);
      const facing=this.direction.dot(v.forward);
      const supported=(1-THREE.MathUtils.smoothstep(distance,3.0,6.5))*
        THREE.MathUtils.smoothstep(facing,-.65,.05);
      if(supported<.001)continue;
      // Ease both endpoints. A nearby view can outweigh a distant one by two
      // orders of magnitude, so a linear first step still looks like a cut.
      u.ramp.value[i]=THREE.MathUtils.smootherstep(layer.fade,0,1);
      const presence=u.ramp.value[i]*supported;
      scanU.fades.value[i]=presence;
      scanU['rgb'+i].value=item.texture;scanU['rangeMap'+i].value=item.depth;
      scanU['pose'+i].value.copy(v.inverse);scanU['lens'+i].value.fromArray(v.lens);
      scanU['distort'+i].value.fromArray(v.distortion);
      layer.target.setSize(w,h);renderer.setRenderTarget(layer.target);
      this.surface.geometry=item.geometry;this.surface.matrix.copy(v.matrix);
      this.surface.matrixWorldNeedsUpdate=true;
      this.surfaceMaterial.uniforms.photo.value=item.texture;this.surfaceMaterial.uniforms.sourceCenter.value.copy(v.position);
      renderer.render(this.layerScene,camera);
      u['layer'+i].value=layer.target.texture;u['depth'+i].value=layer.target.depthTexture;
      u.weights.value[i]=Math.pow((.34+.66*Math.max(0,facing))/(.075+distance*distance),1.45);
      u.fades.value[i]=presence;
    }
    this.projected.setSize(this.size.x,this.size.y);renderer.setRenderTarget(this.projected);
    this.quad.material=this.onScan;renderer.render(this.screen,camera);
    this.quad.material=this.blend;u.scanColor.value=this.projected.texture;
    renderer.setRenderTarget(null);renderer.setClearColor(this.clearColor,oldAlpha);
    renderer.autoClear=false;renderer.render(this.screen,camera);renderer.autoClear=true;
  }
}
