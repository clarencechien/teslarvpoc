import * as THREE from 'three';

// Lift each calibrated photograph onto a continuous local depth surface.
// The small surface grid is shared by movement, station stops and turns.
export function photoSurface(ranges, rays, width, height) {
  const step=3, cols=Math.ceil(width/step), rows=Math.ceil(height/step);
  const samples=new Float32Array(cols*rows);
  const confidence=new Float32Array(samples.length);
  const uv=new Float32Array(samples.length*2);
  const points=new Float32Array(samples.length*3);
  const neighborhood=[];
  for(let row=0;row<rows;row++){
    for(let col=0;col<cols;col++){
      const x=Math.min(width-1,step*col+1),y=Math.min(height-1,step*row+1),i=col+row*cols;
      neighborhood.length=0;
      for(let v=Math.max(y-2,0);v<=Math.min(height-1,y+2);v++){
        for(let u=Math.max(0,x-2);u<=Math.min(width-1,x+2);u++){
          const j=4*(u+v*width),d=(ranges[j]*256+ranges[j+1])*.001;
          if(d>.3 && d<48)neighborhood.push(1/d);
        }
      }
      neighborhood.sort((a,b)=>a-b);
      samples[i]=neighborhood[Math.floor(neighborhood.length/2)]||0;
      confidence[i]=ranges[4*(x+y*width)+2]/255;
      uv[2*i]=(x+.5)/width;uv[2*i+1]=(y+.5)/height;
    }
  }
  // Inverse range changes slowly on large flat surfaces. Interpolate short
  // gaps and limit depth spikes that would fold the projection across itself.
  let depth=samples;
  for(let iteration=0;iteration<3;iteration++){
    const next=new Float32Array(depth.length);
    for(let row=0;row<rows;row++){
      for(let col=0;col<cols;col++){
        const i=row*cols+col;
        let sum=0,weight=0;
        for(let dy=-1;dy<=1;dy++){
          for(let dx=-1;dx<=1;dx++){
            const x=col+dx,y=row+dy;
            if(x<0||x>=cols||y<0||y>=rows)continue;
            const value=depth[y*cols+x];
            if(value<=0)continue;
            const w=(dx===0?2:1)*(dy===0?2:1);
            sum+=w*value;weight+=w;
          }
        }
        next[i]=weight?sum/weight:0;
      }
    }
    depth=next;
  }
  const indices=[];
  for(let row=0;row<rows;row++){
    for(let col=0;col<cols;col++){
      const i=row*cols+col,x=Math.min(width-1,step*col+1),y=Math.min(height-1,step*row+1);
      const range=1/Math.max(1/40,depth[i]);
      for(let k=0;k<3;k++)points[3*i+k]=rays[3*(x+y*width)+k]*range;
      if(col+1<cols && row+1<rows){
        const a=i,b=i+1,c=i+cols,d=c+1;
        if(depth[a]&&depth[b]&&depth[c])indices.push(a,c,b);
        if(depth[b]&&depth[c]&&depth[d])indices.push(b,c,d);
      }
    }
  }
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.BufferAttribute(points,3));
  g.setAttribute('uv',new THREE.BufferAttribute(uv,2));
  g.setAttribute('certainty',new THREE.BufferAttribute(confidence,1));g.setIndex(indices);
  return g;
}
