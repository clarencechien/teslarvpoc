import * as THREE from 'three';

// Each local cloud lives in its anchor camera's coordinates (right, up, back).
// Runtime parallax translates the eye without rotating its depth axis, so the
// far-to-near file ordering stays valid at every supported interactive pose.
export async function loadSplats(id) {
  const response=await fetch('./station-'+id+'.splatbin');
  if(!response.ok) throw Error('實景資料下載失敗');
  const bytes=await response.arrayBuffer();
  if(!bytes.byteLength || bytes.byteLength%12) throw Error('實景資料不完整');
  const data=new DataView(bytes), count=bytes.byteLength/12;
  const position=new Float32Array(count*3), tint=new Uint8Array(count*4), sigma=new Float32Array(count);
  for(let i=0;i<count;i++){
    const base=i*12;
    for(let k=0;k<3;k++)position[3*i+k]=data.getInt16(base+2*k,true)*.001;
    sigma[i]=data.getUint16(base+6,true)*1e-6;
    for(let k=0;k<4;k++)tint[4*i+k]=data.getUint8(base+8+k);
  }
  const geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.BufferAttribute(position,3));
  geometry.setAttribute('radius',new THREE.BufferAttribute(sigma,1));
  geometry.setAttribute('tint',new THREE.BufferAttribute(tint,4,true));
  const material=new THREE.ShaderMaterial({
    uniforms:{rasterHeight:{value:480}},
    vertexShader:`
      attribute float radius; attribute vec4 tint;
      uniform float rasterHeight;
      varying vec4 rgba;
      varying vec3 conic;
      varying float diameter;
      void main(){
        vec4 eye=modelViewMatrix*vec4(position,1.);
        gl_Position=projectionMatrix*eye;
        float z=max(.02,-eye.z);
        float f=projectionMatrix[1][1]*rasterHeight*.5;
        // Project the flattened 3D covariance through the pinhole Jacobian.
        vec2 slope=eye.xy/z;
        float pixelVariance=pow(f*radius/z,2.);
        float a=pixelVariance*(1.+.09*slope.x*slope.x)+.08;
        float b=pixelVariance*.09*slope.x*slope.y;
        float c=pixelVariance*(1.+.09*slope.y*slope.y)+.08;
        conic=vec3(c,-b,a)/max(.00001,a*c-b*b);
        float maximum=.5*(a+c+sqrt((a-c)*(a-c)+4.*b*b));
        diameter=ceil(5.8*sqrt(maximum));
        gl_PointSize=diameter;
        rgba=tint;
      }`,
    fragmentShader:`
      varying vec4 rgba;
      varying vec3 conic;
      varying float diameter;
      void main(){
        vec2 d=(gl_PointCoord-vec2(.5))*diameter;
        d.y=-d.y;
        float weight=exp(-.5*(conic.x*d.x*d.x+2.*conic.y*d.x*d.y+conic.z*d.y*d.y));
        if(weight<.015)discard;
        gl_FragColor=vec4(rgba.rgb,rgba.a*weight);
      }`,
    transparent:true, depthTest:false, depthWrite:false, toneMapped:false
  });
  const cloud=new THREE.Points(geometry,material);
  cloud.frustumCulled=false;
  return cloud;
}

export function localCamera(camera, aspect, horizontal, vertical) {
  // Fit a crop of the source image. The safety margin keeps the UI inside
  // captured rays even at the end of the parallax range.
  const halfHeight=Math.min(215,287/aspect);
  camera.fov=2*THREE.MathUtils.radToDeg(Math.atan(halfHeight/525));
  camera.aspect=aspect;camera.near=.025;camera.far=20;
  camera.position.set(horizontal*.045,vertical*.025,0);
  camera.quaternion.identity();
  camera.updateProjectionMatrix();
}

// Tiles share one atlas and one draw call. Translating the camera moves each
// depth plane by its own disparity; the result remains continuously navigable.
export async function loadSplatLayers(id) {
  const base='./station-'+id;
  const [response,atlas]=await Promise.all([fetch(base+'.json'),new THREE.TextureLoader().loadAsync(base+'.webp')]);
  if(!response.ok)throw Error('實景圖層下載失敗');
  const {size,quads}=await response.json();
  atlas.generateMipmaps=false;atlas.minFilter=THREE.LinearFilter;
  atlas.premultiplyAlpha=true;atlas.flipY=false;
  const positions=[],uvs=[],indices=[];
  for(const [depth,x0,y0,x1,y1,u,v] of quads){
    const baseIndex=positions.length/3;
    for(const [x,y] of [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]){
      positions.push((x-384)*depth/525,(288-y)*depth/525,-depth);
      uvs.push((u+x-x0)/size[0],(v+y-y0)/size[1]);
    }
    indices.push(baseIndex,baseIndex+2,baseIndex+1,baseIndex,baseIndex+3,baseIndex+2);
  }
  const geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
  geometry.setAttribute('uv',new THREE.Float32BufferAttribute(uvs,2));geometry.setIndex(indices);
  const material=new THREE.ShaderMaterial({
    uniforms:{atlas:{value:atlas}},transparent:true,premultipliedAlpha:true,depthTest:false,depthWrite:false,
    vertexShader:'varying vec2 sourceUV;void main(){sourceUV=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}',
    fragmentShader:'varying vec2 sourceUV;uniform sampler2D atlas;void main(){gl_FragColor=texture2D(atlas,sourceUV);}'
  });
  return new THREE.Mesh(geometry,material);
}
