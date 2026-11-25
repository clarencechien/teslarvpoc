import * as THREE from 'three';
import {makePhotoAnchor} from './church-photo-anchor.js';

// Unproject a regular grid through the calibrated equidistant fisheye lens. The
// lookup is shared by all photographs from a camera, while range changes per exposure.
const raysByLens=new Map();
function gridRays(view,width,height) {
  const key=[width,height,...view.lens,...view.distortion].join(',');
  if(raysByLens.has(key)) return raysByLens.get(key);
  const [fx,fy,cx,cy]=view.lens,[k1,k2,k3,k4]=view.distortion;
  const rays=new Float32Array(width*height*3);
  for(let i=0;i<width*height;i++){
    const x=(((2*(i%width)+.5)*1440/(width*2)-.5)-cx)/fx;
    const y=(((2*Math.floor(i/width)+.5)*1080/(height*2)-.5)-cy)/fy;
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

function depthSurface(view,depthImage) {
  const canvas=document.createElement('canvas');
  canvas.width=depthImage.width;canvas.height=depthImage.height;
  const ctx=canvas.getContext('2d',{willReadFrequently:true});
  ctx.drawImage(depthImage,0,0);
  const rgba=ctx.getImageData(0,0,canvas.width,canvas.height).data;
  const w=canvas.width/2,h=canvas.height/2;
  const rays=gridRays(view,w,h),positions=new Float32Array(rays.length),depth=new Float32Array(w*h);
  const indices=[];
  for(let i=0;i<w*h;i++){
    const p=4*(2*(i%w)+2*Math.floor(i/w)*canvas.width);
    depth[i]=(256*rgba[p]+rgba[p+1])*.001;
    for(let j=0;j<3;j++)positions[3*i+j]=rays[3*i+j]*depth[i];
  }
  function triangle(a,b,c) {
    const near=Math.min(depth[a],depth[b],depth[c]),far=Math.max(depth[a],depth[b],depth[c]);
    if(near>.30 && far-near<.075+.037*near)indices.push(a,b,c);
  }
  for(let row=0;row<h-1;row++)for(let col=0;col<w-1;col++){
    const a=row*w+col;triangle(a,a+w,a+1);triangle(a+1,a+w,a+w+1);
  }
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.BufferAttribute(positions,3));g.setIndex(indices);
  return g;
}

const slots=[0,1,2,3];
const vertex=`
  varying vec3 world;
  void main(){
    vec4 p=modelMatrix*vec4(position,1.);
    world=p.xyz;gl_Position=projectionMatrix*viewMatrix*p;
    // Prefer the fitted scan when the two measured surfaces agree to a few centimetres.
    gl_Position.z+=-projectionMatrix[3][2]*.075/max(.001,gl_Position.w+.075);
  }`;

const photoUniforms={};
function projectionCode(){
  const declarations=[];
  for(const i of slots){
    for(const [name,value] of Object.entries({photo:null,rangeMap:null,source:new THREE.Matrix4(),
      lens:new THREE.Vector4(),distortion:new THREE.Vector4(),center:new THREE.Vector3()})){
      photoUniforms[name+i]={value};
    }
    declarations.push(`uniform sampler2D photo${i},rangeMap${i}; uniform mat4 source${i};
      uniform vec3 center${i}; uniform vec4 lens${i},distortion${i};`);
  }
  return `
    ${declarations.join('\n')}
    float support(vec3 packedRange,float distanceToSource){
      float d=dot(floor(packedRange.rg*255.+.5),vec2(256.,1.))*.001;
      return packedRange.b*(1.-smoothstep(.085+.012*d,.22+.026*d,abs(d-distanceToSource)));
    }
    void observation(vec3 world,sampler2D photo,sampler2D ranges,mat4 src,vec4 lens,vec4 k,vec3 center,
                     out vec3 rgb,out vec3 low,out float weight){
      rgb=low=vec3(0.);weight=0.;
      vec3 p=(src*vec4(world,1.)).xyz;
      if(p.z>=-.10)return;
      float r=length(p.xy),theta=atan(r,-p.z),s=theta*theta;
      float radial=theta*(1.+s*(k.x+s*(k.y+s*(k.z+s*k.w))));
      vec2 uv=(vec2(p.x,-p.y)*radial/max(r,.00001)*lens.xy+lens.zw+.5)/vec2(1440.,1080.);
      float edge=min(min(uv.x,1.-uv.x),min(uv.y,1.-uv.y));
      if(edge<=0.)return;
      vec3 packedRange=texture2D(ranges,uv).rgb;
      vec2 pad=1.5/vec2(360.,270.);
      float distanceToSource=length(p);
      float vis=.4*support(packedRange,distanceToSource);
      vis+=.15*(support(texture2D(ranges,uv+vec2(pad.x,0.)).rgb,distanceToSource)+
                 support(texture2D(ranges,uv-vec2(pad.x,0.)).rgb,distanceToSource)+
                 support(texture2D(ranges,uv+vec2(0.,pad.y)).rgb,distanceToSource)+
                 support(texture2D(ranges,uv-vec2(0.,pad.y)).rgb,distanceToSource));
      float baseline=distance(center,cameraPosition);
      vec3 ray=normalize(world-center),targetRay=normalize(world-cameraPosition);
      float parallax=length(ray-targetRay);
      weight=vis*smoothstep(0.,.045,edge)*
        (1.-smoothstep(2.2,4.2,baseline))/(.32+.60*baseline*baseline+15.*parallax*parallax);
      rgb=texture2D(photo,uv).rgb;
      low=textureLod(photo,uv,2.2).rgb;
    }
    vec4 photographicColor(vec3 world){
      vec3 rgb[4],low[4];float w[4];
      ${slots.map(i=>`observation(world,photo${i},rangeMap${i},source${i},lens${i},distortion${i},center${i},rgb[${i}],low[${i}],w[${i}]);`).join('\n')}
      // Choose a sharp observation with support from other sources. Exposure
      // and alignment differences should not superimpose two rail or window edges.
      int best=0;float peak=0.;
      for(int i=0;i<4;i++){
        float agree=.30;
        for(int j=0;j<4;j++){
          if(i!=j)agree+=min(w[j],1.)/(1.+18.*length(low[i]-low[j]));
        }
        float score=w[i]*agree;
        if(score>peak){peak=score;best=i;}
      }
      float sum=0.;vec3 color=vec3(0.);
      for(int i=0;i<4;i++){
        float delta=length(low[i]-low[best]);
        float weight=w[i]/(1.+120.*delta*delta);
        sum+=weight;color+=weight*rgb[i];
      }
      return vec4(max(vec3(0.),color/max(sum,.00001)),smoothstep(.03,.36,sum));
    }`;
}
const photoCode=projectionCode();
function photographicMaterial(){
  const uniforms={...photoUniforms,base:{value:null},zMap:{value:null},
    invProjection:{value:new THREE.Matrix4()},viewPose:{value:new THREE.Matrix4()}};
  return new THREE.ShaderMaterial({uniforms,depthTest:false,depthWrite:false,
    vertexShader:'varying vec2 screen;void main(){screen=position.xy;gl_Position=vec4(position.xy,0.,1.);}',
    fragmentShader:`varying vec2 screen;
      uniform sampler2D base,zMap;
      uniform mat4 invProjection,viewPose;
      ${photoCode}
      void main(){
        vec2 uv=screen*.5+.5;
        float z=texture2D(zMap,uv).r;
        vec3 color=texture2D(base,uv).rgb;
        if(z<.999999){
          vec4 eye=invProjection*vec4(screen,2.*z-1.,1.);
          vec4 photo=photographicColor((viewPose*vec4(eye.xyz/eye.w,1.)).xyz);
          color=mix(color,photo.rgb,photo.a);
        }
        gl_FragColor=vec4(color,1.);
        #include <colorspace_fragment>
      }`});
}

export class PhotoWalk {
  constructor(views,invalidate){
    this.inflight=new Map();
    this.drawAnchor=makePhotoAnchor();
    this.views=views;this.invalidate=invalidate;
    this.scene=new THREE.Scene();
    this.material=photographicMaterial();
    this.screen=new THREE.Scene();
    const quad=new THREE.Mesh(new THREE.PlaneGeometry(2,2),this.material);
    quad.frustumCulled=false;this.screen.add(quad);
    this.size=new THREE.Vector2();
    this.target=new THREE.WebGLRenderTarget(1,1,{depthTexture:new THREE.DepthTexture(1,1,THREE.UnsignedIntType)});
    this.target.texture.colorSpace=THREE.SRGBColorSpace;
    this.depthOnly=new THREE.ShaderMaterial({vertexShader:vertex,fragmentShader:'void main(){gl_FragColor=vec4(0.);}',
      side:THREE.DoubleSide,colorWrite:false,polygonOffset:true,polygonOffsetFactor:1,polygonOffsetUnits:2});
    this.cache=new Map();this.active=[];this.loader=new THREE.TextureLoader();
    this.direction=new THREE.Vector3();this.enabled=false;this.loading=false;this.retryAt=0;
  }
  static async create(views,invalidate) {
    async function load(name){
      const r=await fetch('./photo-walk/'+name+'.json');
      if(!r.ok)throw Error('實拍索引讀取失敗');return r.json();
    }
    const ids=new Set(await load('sources'));
    return new PhotoWalk(views.filter(v=>ids.has(v.id)),invalidate);
  }
  bundle(view){
    if(this.cache.has(view.id))return Promise.resolve(this.cache.get(view.id));
    if(!this.inflight.has(view.id))this.inflight.set(view.id,this.loadBundle(view).finally(()=>this.inflight.delete(view.id)));
    return this.inflight.get(view.id);
  }
  async loadBundle(view){
    const [photo,depth]=await Promise.allSettled([this.loader.loadAsync('./photo-'+view.id+'.webp'),
      this.loader.loadAsync('./photo-walk/'+view.id+'.png')]);
    if(photo.status==='rejected'||depth.status==='rejected'){
      if(photo.status==='fulfilled')photo.value.dispose();
      if(depth.status==='fulfilled')depth.value.dispose();
      throw Error('實拍圖層讀取失敗');
    }
    photo.value.flipY=false;photo.value.colorSpace=THREE.SRGBColorSpace;
    depth.value.flipY=false;depth.value.generateMipmaps=false;
    depth.value.minFilter=depth.value.magFilter=THREE.NearestFilter;
    const mesh=new THREE.Mesh(depthSurface(view,depth.value.image),this.depthOnly);
    mesh.matrixAutoUpdate=false;mesh.matrix.copy(view.matrix);
    const result={view,mesh,photo:photo.value,depth:depth.value};this.cache.set(view.id,result);return result;
  }
  select(camera) {
    if(!this.enabled)return;
    if(this.loading||performance.now()<this.retryAt)return;
    const forward=camera.getWorldDirection(this.direction);
    const scored=this.views.map(view=>({view,score:camera.position.distanceToSquared(view.position)+
      4.2*(1-forward.dot(view.forward))})).sort((a,b)=>a.score-b.score);
    const nearest=scored.slice(0,slots.length).map(entry=>entry.view).sort((a,b)=>a.id-b.id);
    if(!nearest.length||nearest.every((v,i)=>v===this.active[i]?.view))return;
    this.loading=true;
    Promise.allSettled(nearest.map(v=>this.bundle(v))).then(results=>{
      if(results.some(r=>r.status==='rejected')){this.retryAt=performance.now()+1800;return;}
      this.active=results.map(r=>r.value);
      this.scene.clear();
      const u=this.material.uniforms;
      this.active.forEach(({view,mesh,photo,depth},i)=>{
        this.scene.add(mesh);u['photo'+i].value=photo;u['rangeMap'+i].value=depth;
        u['source'+i].value.copy(view.matrix).invert();u['center'+i].value.copy(view.position);
        u['lens'+i].value.fromArray(view.lens);u['distortion'+i].value.fromArray(view.distortion);
      });
      this.invalidate();
    }).finally(()=>{
      this.loading=false;
      for(const [id,item] of this.cache){
        if(this.cache.size<=8)break;
        if(this.active.includes(item))continue;
        item.mesh.geometry.dispose();item.photo.dispose();item.depth.dispose();this.cache.delete(id);
      }
    });
  }
  render(renderer,camera,scan) {
    // Shade the nearest supported surface once per output pixel, regardless of
    // how many of the overlapping measured patches contribute to its depth.
    renderer.getDrawingBufferSize(this.size);
    this.target.setSize(this.size.x,this.size.y);
    renderer.setRenderTarget(this.target);renderer.render(scan,camera);
    renderer.autoClear=false;renderer.render(this.scene,camera);
    renderer.setRenderTarget(null);
    const u=this.material.uniforms;
    u.base.value=this.target.texture;u.zMap.value=this.target.depthTexture;
    u.invProjection.value.copy(camera.projectionMatrixInverse);u.viewPose.value.copy(camera.matrixWorld);
    renderer.render(this.screen,camera);this.drawAnchor(renderer,camera,this.active);
    renderer.autoClear=true;
  }
}
