import * as THREE from 'three';

// At a registered camera center a ray has a known photographic colour even over
// glass, a moving person or a hole in the survey. Nearby, compensate for parallax
// using range, then fade to the multi-view 3D solution as the baseline increases.
export function makePhotoAnchor(){
  const uniforms={invProjection:{value:new THREE.Matrix4()},worldFromEye:{value:new THREE.Matrix4()},
    source:{value:new THREE.Matrix4()},lens:{value:new THREE.Vector4()},k:{value:new THREE.Vector4()},
    offset:{value:new THREE.Vector3()},strength:{value:0},photo:{value:null},ranges:{value:null}};
  const mat=new THREE.ShaderMaterial({uniforms,transparent:true,depthTest:false,depthWrite:false,
    vertexShader:'varying vec2 screen;void main(){screen=position.xy;gl_Position=vec4(position.xy,0.,1.);}',
    fragmentShader:`
      varying vec2 screen;
      uniform mat4 invProjection,worldFromEye,source;
      uniform vec4 lens,k;
      uniform vec3 offset;
      uniform float strength;
      uniform sampler2D photo,ranges;
      vec3 projectRay(vec3 ray){
        vec3 p=mat3(source)*ray;
        float r=length(p.xy),theta=atan(r,-p.z),s=theta*theta;
        float radial=theta*(1.+s*(k.x+s*(k.y+s*(k.z+s*k.w))));
        return vec3((vec2(p.x,-p.y)*radial/max(r,.00001)*lens.xy+lens.zw+.5)/vec2(1440.,1080.),-p.z);
      }
      void main(){
        vec4 p=invProjection*vec4(screen,1.,1.);
        vec3 ray=normalize(mat3(worldFromEye)*p.xyz);
        vec3 uvz=projectRay(ray);float d=0.,error=0.;
        for(int i=0;i<3;i++){
          vec2 bytes=floor(texture2D(ranges,uvz.xy).rg*255.+.5);
          float observed=dot(bytes,vec2(256.,1.))*.001;
          error=max(error,abs(d-observed)*float(i>0));
          d=observed;
          float along=dot(ray,offset);
          float travel=sqrt(max(.01,d*d-dot(offset,offset)+along*along))-along;
          uvz=projectRay(ray*(d>.25?max(travel,.1):8.)+offset);
        }
        float edge=min(min(uvz.x,1.-uvz.x),min(uvz.y,1.-uvz.y));
        float unstable=max(float(d<.25),smoothstep(.16,.48,error))*smoothstep(.01,.16,length(offset));
        float alpha=strength*(1.-unstable)*smoothstep(0.,.025,edge);
        if(uvz.z<=0.||alpha<.001)discard;
        gl_FragColor=vec4(texture2D(photo,uvz.xy).rgb,alpha);
        #include <colorspace_fragment>
      }`});
  const quad=new THREE.Mesh(new THREE.PlaneGeometry(2,2),mat);quad.frustumCulled=false;
  const scene=new THREE.Scene();scene.add(quad);
  const direction=new THREE.Vector3();
  return (renderer,camera,candidates)=>{
    camera.getWorldDirection(direction);
    let best=null,bestScore=Infinity;
    for(const item of candidates){
      const distance=camera.position.distanceTo(item.view.position),facing=direction.dot(item.view.forward);
      if(distance>=.52||facing<.35)continue;
      const score=distance*distance+.19*(1-facing);
      if(score<bestScore){best=item;bestScore=score;}
    }
    if(!best)return;
    const v=best.view;
    uniforms.invProjection.value.copy(camera.projectionMatrixInverse);uniforms.worldFromEye.value.copy(camera.matrixWorld);
    uniforms.source.value.copy(v.matrix).invert();uniforms.lens.value.fromArray(v.lens);uniforms.k.value.fromArray(v.distortion);
    uniforms.offset.value.subVectors(camera.position,v.position);
    uniforms.strength.value=1.-THREE.MathUtils.smoothstep(uniforms.offset.value.length(),.025,.52);
    uniforms.photo.value=best.photo;uniforms.ranges.value=best.depth;renderer.render(scene,camera);
  };
}
