'use strict';
const el = id => document.getElementById(id);
const model = JSON.parse(el('model').textContent);
el('download').onclick = () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(model)], {type:'model/gltf+json'}));
  const link = document.createElement('a');
  link.href = url;
  link.download = 'middlebury-motorcycle.gltf';
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
};
const canvas = document.querySelector('canvas');
const gl = canvas.getContext('webgl', {antialias:false, alpha:false});
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

function shader(type, source) {
  const id = gl.createShader(type);
  gl.shaderSource(id, source);
  gl.compileShader(id);
  if (!gl.getShaderParameter(id, gl.COMPILE_STATUS)) throw Error(gl.getShaderInfoLog(id));
  return id;
}

async function start() {
  if (!gl) throw Error('此瀏覽器無法開啟 WebGL。可下載模型至 Blender 檢視。');
  const program = gl.createProgram();
  gl.attachShader(program, shader(gl.VERTEX_SHADER, `
    attribute vec3 point;
    attribute vec2 uv;
    uniform vec2 angle, scale;
    uniform float radius;
    varying mediump vec2 texcoord;
    varying mediump float depth;
    void main(){
      vec3 p = point + vec3(0.,0.,2.1);
      vec2 s=sin(angle), c=cos(angle);
      float z = s.x*p.x+c.x*p.z;
      vec3 view=vec3(c.x*p.x-s.x*p.z,c.y*p.y+s.y*z,-s.y*p.y+c.y*z-radius);
      gl_Position=vec4(scale*view.xy,-1.01*view.z-0.10,-view.z);
      gl_PointSize=2.;
      texcoord=uv;
      depth=-point.z;
    }`));
  gl.attachShader(program, shader(gl.FRAGMENT_SHADER, `
    precision mediump float;
    uniform sampler2D photo;
    uniform int mode;
    varying vec2 texcoord;
    varying float depth;
    void main(){
      vec3 col=texture2D(photo,texcoord).rgb;
      if(mode==1) col=mix(vec3(.99,.69,.31),vec3(.10,.31,.49),clamp((depth-1.4)/2.4,0.,1.));
      gl_FragColor=vec4(col,1.);
    }`));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw Error(gl.getProgramInfoLog(program));
  gl.useProgram(program);
  const encoded = model.buffers[0].uri.split(',')[1];
  const bytes = Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
  function bindAccessor(index, target, attribute, size) {
    const acc = model.accessors[index], view = model.bufferViews[acc.bufferView];
    gl.bindBuffer(target, gl.createBuffer());
    gl.bufferData(target, bytes.subarray(view.byteOffset, view.byteOffset+view.byteLength), gl.STATIC_DRAW);
    if (attribute) {
      const location = gl.getAttribLocation(program, attribute);
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, size, gl.FLOAT, false, 0, 0);
    }
    return acc.count;
  }
  const primitive = model.meshes[0].primitives[0];
  const points = bindAccessor(primitive.attributes.POSITION,gl.ARRAY_BUFFER,'point',3);
  bindAccessor(primitive.attributes.TEXCOORD_0,gl.ARRAY_BUFFER,'uv',2);
  const indices = bindAccessor(primitive.indices,gl.ELEMENT_ARRAY_BUFFER);
  const img = new Image();
  img.src = model.images[0].uri;
  await img.decode();
  gl.bindTexture(gl.TEXTURE_2D,gl.createTexture());
  for (const edge of [gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D,edge,gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGB,gl.RGB,gl.UNSIGNED_BYTE,img);
  gl.enable(gl.DEPTH_TEST);
  gl.clearColor(.10,.13,.14,1);
  const u = Object.fromEntries(['angle','scale','radius','mode'].map(key=>[key,gl.getUniformLocation(program,key)]));
  el('mesh').textContent = (indices/3).toLocaleString()+' 三角面';
  let yaw=.06, pitch=0, radius=2.1, drag=null, bench=null;
  function rotate(dx,dy) {
    yaw=clamp(yaw+dx,-1.3,1.3);
    pitch=clamp(pitch+dy,-.85,.85);
  }
  function zoom(factor) {radius=clamp(radius*factor,1.15,4.5);}
  canvas.onpointerdown=e=>{
    if (bench || !e.isPrimary) return;
    canvas.setPointerCapture(e.pointerId);
    drag=[e.clientX,e.clientY];
  };
  canvas.onpointermove=e=>{
    if (!drag || !e.isPrimary) return;
    rotate((drag[0]-e.clientX)*.006,(e.clientY-drag[1])*.006);
    drag=[e.clientX,e.clientY];
  };
  canvas.onlostpointercapture=()=>{drag=null;};
  canvas.addEventListener('wheel',e=>{e.preventDefault();if(!bench) zoom(Math.exp(e.deltaY*.001));},{passive:false});
  canvas.onkeydown=e=>{
    if(bench) return;
    switch(e.key) {
      case 'ArrowLeft': rotate(-.06,0); break;
      case 'ArrowRight': rotate(.06,0); break;
      case 'ArrowUp': rotate(0,.06); break;
      case 'ArrowDown': rotate(0,-.06); break;
      case '+': case '=': zoom(.9); break;
      case '-': zoom(1.1); break;
      default: return;
    }
    e.preventDefault();
  };
  el('reset').onclick=()=>{yaw=0;pitch=0;radius=2.1;};
  function finish(message) {
    el('result').textContent=message;
    el('resolution').disabled=el('mode').disabled=el('bench').disabled=el('reset').disabled=false;
    bench=null;
  }
  el('bench').onclick=()=>{
    drag=null;
    bench={start:null,last:null,intervals:[],width:canvas.width,height:canvas.height};
    el('mode').value='0';
    el('resolution').disabled=el('mode').disabled=el('bench').disabled=el('reset').disabled=true;
    el('result').textContent='正在以固定路徑旋轉模型。暖機 2 秒後量測 10 秒，請保持此分頁在前景。';
  };
  document.addEventListener('visibilitychange',()=>{
    if (bench && document.hidden) finish('測試已中止，請保持分頁在前景後重新測試。');
  });
  let since=null, frames=0;
  function render(now) {
    const ratio=Math.min(window.devicePixelRatio || 1,1.5)*Number(el('resolution').value);
    const w=Math.max(1,Math.round(canvas.clientWidth*ratio)), h=Math.max(1,Math.round(canvas.clientHeight*ratio));
    if(canvas.width!==w || canvas.height!==h) {
      canvas.width=w; canvas.height=h; gl.viewport(0,0,w,h);
    }
    let x=yaw,y=pitch,r=radius;
    if(bench && (w!==bench.width || h!==bench.height)) finish('畫布尺寸已改變，請重新測試。');
    if(bench) {
      bench.start ??= now;
      const elapsed=now-bench.start;
      x=.26*Math.sin(elapsed/1400); y=.055*Math.sin(elapsed/1000); r=2.1;
      if(elapsed>=2000) {
        if(bench.last!==null) bench.intervals.push(now-bench.last);
        bench.last=now;
      }
      if(elapsed>=12000 && bench.intervals.length) {
        const intervals=bench.intervals.sort((a,b)=>a-b);
        const fps=intervals.length*1000/intervals.reduce((a,b)=>a+b,0);
        const p95=intervals[Math.ceil(intervals.length*.95)-1];
        finish(w+' × '+h+' px · '+intervals.length+' 個畫格間隔 · 平均 '+fps.toFixed(1)+' FPS · P95 '+p95.toFixed(1)+' ms');
      }
    }
    const aspect=w/h, sy=Math.min(3.4,2.40*aspect);
    gl.uniform2f(u.angle,x,y); gl.uniform2f(u.scale,sy/aspect,sy); gl.uniform1f(u.radius,r);
    const mode=Number(el('mode').value);
    gl.uniform1i(u.mode,mode);
    gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    if(mode===2) gl.drawArrays(gl.POINTS,0,points);
    else gl.drawElements(gl.TRIANGLES,indices,gl.UNSIGNED_SHORT,0);
    since ??= now;
    if(now-since>=1000) {
      el('fps').textContent=(1000*frames/(now-since)).toFixed(0)+' FPS';
      since=now; frames=0;
    }
    frames++;
    requestAnimationFrame(render);
  }
  requestAnimationFrame(render);
}

start().catch(err=>{
  el('result').textContent=err.message;
  el('bench').disabled=el('mode').disabled=el('reset').disabled=true;
});
