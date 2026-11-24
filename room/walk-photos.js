import * as THREE from 'three';

// Project onto the existing surface only where the source camera actually saw it.
// Measured depth is packed into alpha in 1/32 m steps.
const samplePhoto = `
varying vec3 photoWorld;
uniform sampler2D firstPhoto, nextPhoto, supportPhoto;
uniform mat4 firstPose, nextPose, supportPose;
uniform float firstStrength, nextStrength, supportStrength, transition, supportMix;
vec4 projectPhoto(sampler2D photo, mat4 pose, float strength) {
  if (strength <= 0.0) return vec4(0.0);
  vec3 p = (pose * vec4(photoWorld, 1.0)).xyz;
  if (p.z < 0.15) return vec4(0.0);
  vec2 uv = vec2(0.5) + 525.0 * p.xy / (p.z * vec2(640.0,480.0));
  vec2 margin = min(uv, 1.0 - uv);
  if (min(margin.x, margin.y) <= 0.0) return vec4(0.0);
  vec4 sampleColor = texture2D(photo, uv);
  // RGB can be observed even when the depth sensor returns no range. At an
  // almost coincident optical centre, projection approaches pure rotation.
  float edge = smoothstep(0.0, 0.065, min(margin.x, margin.y));
  if (sampleColor.a > 254.5/255.0) {
    float baseline = length((pose * vec4(cameraPosition,1.0)).xyz);
    float nearSource = 1.0-smoothstep(.006,.055,baseline/max(p.z,.45));
    return vec4(sampleColor.rgb, strength * edge * nearSource);
  }
  float sourceZ = (255.0 - 255.0 * sampleColor.a) / 32.0;
  float error = abs(sourceZ - p.z);
  float tolerance = 0.028 + 0.018 * p.z;
  float coverage = edge *
                   (1.0-smoothstep(tolerance, 2.5*tolerance, error));
  return vec4(sampleColor.rgb, strength * coverage * step(0.08,sampleColor.a));
}
`;

function uniform(value) { return {value}; }

export class WalkPhotos {
  constructor(base = './walk-photos/') {
    this.base = base;
    this.loader = new THREE.TextureLoader();
    this.cache = new Map();
    this.direction = new THREE.Vector3();
    this.lastDirection = new THREE.Vector3();
    this.lastPosition = new THREE.Vector3();
    this.recheckAt = -Infinity;
    this.uniforms = {
      firstPhoto: uniform(null), nextPhoto: uniform(null),
      firstPose: uniform(new THREE.Matrix4()), nextPose: uniform(new THREE.Matrix4()),
      firstStrength: uniform(0), nextStrength: uniform(0), transition: uniform(0),
      supportPhoto: uniform(null), supportPose: uniform(new THREE.Matrix4()),
      supportStrength: uniform(0), supportMix: uniform(0)
    };
    this.support = null;
    this.supportSince = -Infinity;
    this.views = [];
    this.previous = null;
    this.current = null;
    this.pending = null;
    this.changedAt = -Infinity;
    this.lastMix = 1;
  }

  async load() {
    this.catalog ??= fetch(this.base + 'views.json?v=3').then(async response => {
      if (!response.ok) throw Error('Photo catalog unavailable');
      this.views = (await response.json()).map(v => ({
        file: v.file, position: new THREE.Vector3().fromArray(v.center),
        direction: new THREE.Vector3().fromArray(v.forward),
        matrix: new THREE.Matrix4().fromArray(v.matrix)
      }));
    }).catch(() => { this.catalog = null; });
    return this.catalog;
  }

  material(original) {
    const material = original.clone();
    material.onBeforeCompile = shader => {
      Object.assign(shader.uniforms, this.uniforms);
      shader.vertexShader = 'varying vec3 photoWorld;\n' + shader.vertexShader.replace(
        '#include <project_vertex>', '#include <project_vertex>\nphotoWorld = (modelMatrix * vec4(transformed,1.0)).xyz;');
      shader.fragmentShader = samplePhoto + shader.fragmentShader.replace('#include <map_fragment>', `
        #include <map_fragment>
        vec4 a = projectPhoto(firstPhoto, firstPose, firstStrength);
        vec4 b = projectPhoto(nextPhoto, nextPose, nextStrength);
        vec4 c = projectPhoto(supportPhoto, supportPose, supportStrength);
        vec3 base = mix(diffuseColor.rgb, c.rgb, c.a);
        vec3 primary = mix(base,b.rgb,b.a);
        primary = mix(primary,c.rgb,b.a*c.a*supportMix);
        diffuseColor.rgb = mix(mix(base,a.rgb,a.a), primary, transition);
      `);
    };
    return material;
  }

  score(view, camera) {
    const alignment = this.direction.dot(view.direction);
    if (alignment < .38) return Infinity;
    return camera.position.distanceToSquared(view.position) + 2.1 * (1 - Math.min(1,alignment));
  }

  async request(view) {
    if (this.pending || !view || this.cache.has(view)) return;
    // At most one request in flight while the visitor turns. Stale loads can
    // enter the bounded cache but never override the current camera choice.
    if (performance.now() < (view.retryAt ?? 0)) return;
    this.pending = view;
    try {
      const texture = await this.loader.loadAsync(this.base + view.file);
      texture.flipY = false;
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.generateMipmaps = false;
      texture.minFilter = THREE.LinearFilter;
      this.cache.set(view, texture);
      if (this.cache.size > 6) {
        const victim = [...this.cache.keys()].find(v => v !== view && v !== this.current && v !== this.previous && v !== this.support);
        this.cache.get(victim).dispose();
        this.cache.delete(victim);
      }
    } catch {
      view.retryAt = performance.now() + 15000;
    } finally {
      this.pending = null;
      this.recheckAt = -Infinity;
    }
  }

  update(camera, now) {
    if (!this.views.length) return false;
    camera.getWorldDirection(this.direction);
    if (this.lastMix === 1 && now-this.supportSince>260 && now < this.recheckAt &&
        this.lastPosition.equals(camera.position) && this.lastDirection.equals(this.direction)) return false;
    this.recheckAt = now + 1000;
    this.lastPosition.copy(camera.position);
    this.lastDirection.copy(this.direction);
    let best = this.current, cost = best ? this.score(best, camera) * .90 : Infinity;
    let second = null, secondCost = Infinity;
    for (const view of this.views) {
      if (view === this.current) continue;
      const score = this.score(view, camera);
      if (score < cost) {
        second = best;
        secondCost = cost;
        best = view;
        cost = score;
      } else if (score < secondCost) {
        second = view;
        secondCost = score;
      }
    }
    if (cost > 2.8) best = null;
    if (best && !this.cache.has(best)) this.request(best);
    else if (second && secondCost < 1.8 && !this.cache.has(second)) this.request(second);
    const fading = now - this.changedAt < 230;
    let changed = false;
    if ((!best || this.cache.has(best)) && best !== this.current && !fading) {
      this.previous = this.current;
      this.current = best;
      this.changedAt = now;
      changed = true;
    }
    // Prefer the primary source on agreement and fall back pixel by pixel
    // where only the second photo sees the surface.
    const support = second && secondCost < 1.8 && this.cache.has(second) ? second : null;
    if (support !== this.support) {
      this.support = support;
      this.supportSince = now;
      changed = true;
    }
    const ramp = Math.min(1, (now - this.supportSince) / 230);
    const supportStrength = support ? .90 * (1 - THREE.MathUtils.smoothstep(secondCost, .12, 1.8)) * ramp : 0;
    const supportMix = support ? .45 * Math.min(1, ((cost + .006) / (secondCost + .006)) ** 1.5) : 0;
    changed ||= this.uniforms.supportStrength.value !== supportStrength || this.uniforms.supportMix.value !== supportMix;
    this.uniforms.supportStrength.value = supportStrength;
    this.uniforms.supportMix.value = supportMix;
    this.uniforms.supportPhoto.value = support ? this.cache.get(support) : null;
    if (support) this.uniforms.supportPose.value.copy(support.matrix);
    const mix = Math.min(1, (now - this.changedAt) / 230);
    changed ||= this.lastMix !== mix;
    this.lastMix = mix;
    this.uniforms.transition.value = mix;
    for (const [prefix, view] of [['first', mix < 1 ? this.previous : null], ['next', this.current]]) {
      const strength = view ? .98 * (1 - THREE.MathUtils.smoothstep(this.score(view,camera), .12, 2.8)) : 0;
      const handle = this.uniforms[prefix + 'Strength'];
      changed ||= strength !== handle.value;
      handle.value = strength;
      this.uniforms[prefix + 'Photo'].value = view ? this.cache.get(view) : null;
      if (view) this.uniforms[prefix + 'Pose'].value.copy(view.matrix);
    }
    return changed;
  }
}
