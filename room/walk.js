import * as THREE from 'three';

// A first-person camera in the same metric coordinate system as the GLB.
// Both touch and keyboard motion use the floor/clearance mask baked from it.
export class RoomWalk {
  constructor(camera, grid, canvas, buttons, invalidate) {
    this.camera=camera;this.grid=grid;this.invalidate=invalidate;
    this.down=new Set();this.enabled=false;this.drag=null;
    this.last=null;this.position=new THREE.Vector3();
    this.yaw=0;this.pitch=0;
    const keys={KeyW:'forward',ArrowUp:'forward',KeyS:'back',ArrowDown:'back',
      KeyA:'left',ArrowLeft:'left',KeyD:'right',ArrowRight:'right'};
    canvas.addEventListener('keydown',e=>{
      if(!this.enabled||!keys[e.code])return;
      e.preventDefault();this.down.add(keys[e.code]);
    });
    window.addEventListener('keyup',e=>this.down.delete(keys[e.code]));
    window.addEventListener('blur',()=>this.clear());
    canvas.addEventListener('blur',()=>this.down.clear());
    document.addEventListener('visibilitychange',()=>{if(document.hidden)this.clear();});
    canvas.addEventListener('pointerdown',e=>{
      if(!this.enabled||!e.isPrimary)return;
      this.drag={x:e.clientX,y:e.clientY,id:e.pointerId};
      canvas.focus({preventScroll:true});canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove',e=>{
      if(!this.enabled||!this.drag||this.drag.id!==e.pointerId)return;
      this.yaw-=(e.clientX-this.drag.x)*.004;
      this.pitch=THREE.MathUtils.clamp(this.pitch-(e.clientY-this.drag.y)*.004,-1.35,1.35);
      this.drag.x=e.clientX;this.drag.y=e.clientY;this.apply();
    });
    canvas.addEventListener('lostpointercapture',()=>{this.drag=null;});
    for(const button of buttons.querySelectorAll('[data-move]')){
      const move=button.dataset.move;
      button.addEventListener('pointerdown',e=>{
        if(!this.enabled)return;
        e.preventDefault();button.setPointerCapture(e.pointerId);
        this.down.add(move);
      });
      button.addEventListener('lostpointercapture',()=>this.down.delete(move));
      button.addEventListener('click',e=>{
        if(this.enabled && e.detail===0){this.down.add(move);this.advance(.16);this.down.delete(move);}
      });
    }
    this.buttons=buttons;
  }
  clear(){this.down.clear();this.drag=null;this.last=null;}
  setEnabled(value){
    this.enabled=value;this.buttons.hidden=!value;this.clear();
  }
  available(x,z){
    const {origin,step,rows}=this.grid;
    return rows[Math.round((z-origin[1])/step)]?.[Math.round((x-origin[0])/step)]==='1';
  }
  enter(x=1.89,z=-1.10,lookX=.37,lookZ=1.03){
    let best=Infinity,px=x,pz=z;
    const {rows,origin,step}=this.grid;
    for(let j=0;j<rows.length;j++)for(let i=0;i<rows[j].length;i++){
      if(rows[j][i]!=='1')continue;
      const gx=origin[0]+i*step,gz=origin[1]+j*step,dist=(gx-x)**2+(gz-z)**2;
      if(dist<best){best=dist;px=gx;pz=gz;}
    }
    this.position.set(px,1.52,pz);
    this.yaw=Math.atan2(px-lookX,pz-lookZ);this.pitch=-.34;
    this.camera.fov=68;this.camera.updateProjectionMatrix();this.apply();
  }
  apply(){
    this.camera.position.copy(this.position);
    this.camera.rotation.set(this.pitch,this.yaw,0,'YXZ');this.invalidate();
  }
  advance(dt){
    let forward=Number(this.down.has('forward'))-Number(this.down.has('back'));
    let side=Number(this.down.has('right'))-Number(this.down.has('left'));
    const scale=1.05*dt/Math.max(1,Math.hypot(forward,side));
    const dx=(side*Math.cos(this.yaw)-forward*Math.sin(this.yaw))*scale;
    const dz=(-side*Math.sin(this.yaw)-forward*Math.cos(this.yaw))*scale;
    // Substeps prevent tunnelling across narrow obstacles at low frame rates.
    const n=Math.max(1,Math.ceil(Math.hypot(dx,dz)/(this.grid.step*.45)));
    for(let k=0;k<n;k++){
      if(this.available(this.position.x+dx/n,this.position.z))this.position.x+=dx/n;
      if(this.available(this.position.x,this.position.z+dz/n))this.position.z+=dz/n;
    }
    this.apply();
  }
  tick(now){
    if(!this.enabled)return;
    const dt=this.last===null?0:Math.min(.12,(now-this.last)/1000);
    this.last=now;
    if(this.down.size)this.advance(dt);
  }
}
