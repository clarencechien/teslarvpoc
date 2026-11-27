import * as THREE from 'three';

// The scan provides a second geometric observation at occluding edges. Only
// project a photograph onto it when the range measured from that camera agrees.
export function scanProjection(ids) {
  const uniforms={base:{value:null},zMap:{value:null},eye:{value:new THREE.Matrix4()},
    fades:{value:new Float32Array(ids.length)}};
  for(const i of ids){
    for(const [key,value] of Object.entries({rgb:null,rangeMap:null,pose:new THREE.Matrix4(),
      lens:new THREE.Vector4(),distort:new THREE.Vector4()}))uniforms[key+i]={value};
  }
  const material=new THREE.ShaderMaterial({uniforms,depthWrite:false,depthTest:false,
    vertexShader:'varying vec2 st;void main(){st=position.xy*.5+.5;gl_Position=vec4(position.xy,0.,1.);}',
    fragmentShader:`
      varying vec2 st;uniform sampler2D base,zMap;uniform mat4 eye;uniform float fades[${ids.length}];
      ${ids.map(i=>'uniform sampler2D rgb'+i+',rangeMap'+i+';uniform mat4 pose'+i+';uniform vec4 lens'+i+',distort'+i+';').join('\n')}
      vec4 projectSource(vec3 world,sampler2D photo,sampler2D ranges,mat4 source,vec4 lens,vec4 k,float fade){
        if(fade<=0.)return vec4(0.);
        vec3 p=(source*vec4(world,1.)).xyz;
        if(p.z>=-.1)return vec4(0.);
        float r=length(p.xy),theta=atan(r,-p.z),s=theta*theta;
        float radius=theta*(1.+s*(k.x+s*(k.y+s*(k.z+s*k.w))));
        vec2 uv=(vec2(p.x,-p.y)*radius/max(.00001,r)*lens.xy+lens.zw+.5)/vec2(1440.,1080.);
        float edge=min(min(uv.x,1.-uv.x),min(uv.y,1.-uv.y));
        if(edge<=0.)return vec4(0.);
        vec3 observation=texture2D(ranges,uv).rgb;
        float range=dot(floor(observation.rg*255.+.5),vec2(256.,1.))*.001;
        float vis=observation.b*(1.-smoothstep(.08+.012*range,.21+.026*range,abs(range-length(p))));
        vec3 travel=(source*vec4(cameraPosition,1.)).xyz;
        float parallax=length(normalize(p)-normalize(p-travel));
        float w=fade*vis*smoothstep(0.,.05,edge)/(0.25+dot(travel,travel)+14.*parallax*parallax);
        return vec4(texture2D(photo,uv).rgb*w,w);
      }
      void main(){
        float depth=texture2D(zMap,st).r;
        vec4 p=eye*vec4(st*2.-1.,2.*depth-1.,1.);
        vec4 obs=vec4(0.);
        if(depth<.999999){
          vec3 world=p.xyz/p.w;
          ${ids.map(i=>`obs+=projectSource(world,rgb${i},rangeMap${i},pose${i},lens${i},distort${i},fades[${i}]);`).join('\n')}
        }
        float support=smoothstep(.025,.35,obs.a);
        gl_FragColor=vec4(mix(texture2D(base,st).rgb,obs.rgb/max(.00001,obs.a),support),support);
        #include <colorspace_fragment>
      }`});
  return material;
}
