// Rasterize the UV chart triangles once, then integrate camera samples in surface space.
#include <omp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
using namespace std;
namespace fs = std::filesystem;
constexpr int S = 4096, W = 640, H = 480;
using Vec3 = array<float, 3>;
using Vec2 = array<float, 2>;
template <class T>
void read(ifstream& f, T* out, size_t count) {
  if (!f.read(reinterpret_cast<char*>(out), sizeof(T) * count))
    throw runtime_error("Truncated input");
}
template <class T>
void save(const fs::path& path, const vector<T>& v) {
  ofstream f(path, ios::binary);
  if (!f.write(reinterpret_cast<const char*>(v.data()), sizeof(T) * v.size()))
    throw runtime_error("Write failed");
}
Vec3 sub(Vec3 a, Vec3 b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Vec3 unitNormal(Vec3 a, Vec3 b) {
  Vec3 n = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
  float len = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  for (auto& x : n) x /= max(1e-12f, len);
  return n;
}
struct Texel {
  Vec3 p, n;
  int pixel;
};
struct Bin {
  Vec3 center;
  vector<Texel> samples;
};
struct Surface {
  vector<Bin> bins;
  explicit Surface(const fs::path& file) : bins(32 * 26 * 28) {
    ifstream f(file, ios::binary);
    uint32_t nv, nt;
    read(f, &nv, 1);
    read(f, &nt, 1);
    vector<Vec3> pos(nv);
    vector<Vec2> uv(nv);
    vector<array<uint32_t, 3>> faces(nt);
    read(f, pos.data(), nv);
    read(f, uv.data(), nv);
    read(f, faces.data(), nt);
    vector<uint8_t> claimed(S * S);
    for (auto& tri : faces) {
      auto [a, b, c] = tri;
      Vec2 q = uv[a];
      q[0] *= S;
      q[1] *= S;
      float ux = (uv[b][0] - uv[a][0]) * S, uy = (uv[b][1] - uv[a][1]) * S;
      float vx = (uv[c][0] - uv[a][0]) * S, vy = (uv[c][1] - uv[a][1]) * S;
      float det = ux * vy - uy * vx;
      if (abs(det) < 1e-7) continue;
      auto e = sub(pos[b], pos[a]), g = sub(pos[c], pos[a]), n = unitNormal(e, g);
      int loX = max(0, int(floor(q[0] + min({0.f, ux, vx})))),
          hiX = min(S - 1, int(ceil(q[0] + max({0.f, ux, vx}))));
      int loY = max(0, int(floor(q[1] + min({0.f, uy, vy})))),
          hiY = min(S - 1, int(ceil(q[1] + max({0.f, uy, vy}))));
      for (int y = loY; y <= hiY; y++)
        for (int x = loX; x <= hiX; x++) {
          float dx = x + .5f - q[0], dy = y + .5f - q[1];
          float s = (vy * dx - vx * dy) / det, t = (ux * dy - uy * dx) / det;
          if (s < 0 || t < 0 || s + t > 1 || claimed[y * S + x]) continue;
          Vec3 p;
          for (int k = 0; k < 3; k++) p[k] = pos[a][k] + s * e[k] + t * g[k];
          int ix = clamp(int((p[0] + 3) * 4), 0, 31), iy = clamp(int((p[1] + 1) * 4), 0, 25),
              iz = clamp(int((p[2] + 3) * 4), 0, 27);
          auto& bin = bins[(ix * 26 + iy) * 28 + iz];
          bin.center = {(ix + .5f) / 4 - 3, (iy + .5f) / 4 - 1, (iz + .5f) / 4 - 3};
          bin.samples.push_back({p, n, y * S + x});
          claimed[y * S + x] = 1;
        }
    }
    bins.erase(remove_if(bins.begin(), bins.end(), [](auto& b) { return b.samples.empty(); }),
               bins.end());
  }
};
struct View {
  float m[12];
  vector<uint8_t> rgb = vector<uint8_t>(W * H * 3);
  vector<uint16_t> depth = vector<uint16_t>(W * H);
  vector<float> surface = vector<float>(W * H / 4);
  void load(ifstream& f) {
    read(f, m, 12);
    read(f, rgb.data(), rgb.size());
    read(f, depth.data(), depth.size());
    read(f, surface.data(), surface.size());
  }
  // Estimate focus from the ratio of fine to coarse gradients across image tiles.
  // Interpolating a low-resolution map keeps camera preferences coherent over a surface.
  static constexpr int tile = 32, cols = W / tile, rows = H / tile;
  array<float, cols * rows> focus{};
  void measureFocus() {
    for (int ty = 0; ty < rows; ++ty) {
      for (int tx = 0; tx < cols; ++tx) {
        double fine = 0, coarse = 0;
        for (int y = max(3, ty * tile); y < min(H - 3, (ty + 1) * tile); y += 2) {
          for (int x = max(3, tx * tile); x < min(W - 3, (tx + 1) * tile); x += 2) {
            int i = (y * W + x) * 3 + 1;
            for (int stride : {3, W * 3}) {
              fine += abs(rgb[i - stride] - rgb[i + stride]);
              coarse += abs(rgb[i - 3 * stride] - rgb[i + 3 * stride]);
            }
          }
        }
        // The prior keeps flat surfaces and sensor noise from dominating view selection.
        focus[ty * cols + tx] = clamp(float((fine + 320) / (coarse + 640)), .27f, .75f);
      }
    }
  }
  float focusAt(float u, float v) const {
    float x = clamp(u / tile - .5f, 0.f, float(cols - 1)),
          y = clamp(v / tile - .5f, 0.f, float(rows - 1));
    int x0 = int(x), y0 = int(y), x1 = min(cols - 1, x0 + 1), y1 = min(rows - 1, y0 + 1);
    float fx = x - x0, fy = y - y0;
    return (1 - fy) * ((1 - fx) * focus[y0 * cols + x0] + fx * focus[y0 * cols + x1]) +
           fy * ((1 - fx) * focus[y1 * cols + x0] + fx * focus[y1 * cols + x1]);
  }
  Vec3 camera(Vec3 p) const {
    float x = p[0] - m[0], y = p[1] - m[1], z = p[2] - m[2];
    return {x * m[3] + y * m[6] + z * m[9], x * m[4] + y * m[7] + z * m[10],
            x * m[5] + y * m[8] + z * m[11]};
  }
  bool visible(Vec3 p) const {
    auto [x, y, z] = camera(p);
    return z > -.23f && abs(x) < .61f * z + .31f && abs(y) < .46f * z + .31f;
  }
  float sample(const Texel& s, Vec3& col, float* detail = nullptr) const {
    auto [x, y, z] = camera(s.p);
    if (z < .18f) return 0;
    float u = 319.5f + 525.f * x / z, v = 239.5f + 525.f * y / z;
    if (u < 2 || v < 2 || u > 637 || v > 477) return 0;
    int px = int(u + .5f), py = int(v + .5f), id = py * W + px;
    float front = surface[(py / 2) * (W / 2) + px / 2];
    float tolerance = .028f + .014f * z;
    if (!isfinite(front) || abs(z - front) > tolerance * 2.5f) return 0;
    float d = depth[id] * .001f;
    // Allow missing range on windows/walls when the completed surface is
    // the first ray hit. A nearer measured object always occludes the ray.
    if (d > 0 && d < z - tolerance * 2.8f) return 0;
    float mismatch = d > 0 ? abs(d - z) : .095f;
    float depthWeight = .1f + .9f / (1 + powf(mismatch / tolerance, 2));
    float facing =
        ((m[0] - s.p[0]) * s.n[0] + (m[1] - s.p[1]) * s.n[1] + (m[2] - s.p[2]) * s.n[2]) / z;
    if (facing < .045f) return 0;
    float border = min({u, v, 639 - u, 479 - v});
    float weight =
        min(1.f, border / 34) * min(1.f, facing * facing) * depthWeight / (.45f + z * z * .50f);
    if (weight < .001f) return 0;
    int ix = int(u), iy = int(v);
    float fx = u - ix, fy = v - iy;
    for (int k = 0; k < 3; k++) {
      int b = (iy * W + ix) * 3 + k;
      col[k] = (1 - fy) * ((1 - fx) * rgb[b] + fx * rgb[b + 3]) +
               fy * ((1 - fx) * rgb[b + 3 * W] + fx * rgb[b + 3 * W + 3]);
    }
    if (detail) {
      // Prefer pixels with smaller projected footprints and focused source detail.
      *detail = weight * (.70f + focusAt(u, v)) / (.65f + z);
    }
    return weight;
  }
};
struct Match {
  int id;
  float logL, weight;
};
float median(vector<float>& v) {
  if (v.empty()) return 0;
  auto mid = v.begin() + v.size() / 2;
  nth_element(v.begin(), mid, v.end());
  return *mid;
}
vector<float> exposure(const Surface& surf, const fs::path& file) {
  ifstream f(file, ios::binary);
  int count;
  read(f, &count, 1);
  View v;
  vector<vector<Match>> match(count);
  int refs = (S * S + 255) / 256;
  for (int i = 0; i < count; i++) {
    v.load(f);
    for (const auto& b : surf.bins)
      if (v.visible(b.center))
        for (const auto& s : b.samples)
          if (s.pixel % 256 == 0) {
            Vec3 c;
            float w = v.sample(s, c);
            if (w < .04f) continue;
            float l = .114f * c[0] + .587f * c[1] + .299f * c[2];
            if (l > 12 && l < 241) match[i].push_back({s.pixel / 256, log(l), w});
          }
  }
  vector<float> gain(count, 0);
  for (int step = 0; step < 8; step++) {
    vector<float> reference(refs), weights(refs);
    for (int i = 0; i < count; i++)
      for (const auto& obs : match[i]) {
        reference[obs.id] += obs.weight * (obs.logL + gain[i]);
        weights[obs.id] += obs.weight;
      }
    for (int i = 0; i < count; i++) {
      vector<float> delta;
      for (const auto& obs : match[i])
        if (weights[obs.id] > .25f) delta.push_back(reference[obs.id] / weights[obs.id] - obs.logL);
      if (delta.size() > 16) gain[i] = clamp(median(delta), -.63f, .63f);
    }
  }
  vector<float> tmp = gain;
  float center = median(tmp);
  for (auto& x : gain) x = exp(x - center);
  return gain;
}
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  fs::path work = argv[1];
  omp_set_num_threads(4);
  Surface surf(work / "geometry.bin");
  const auto gains = exposure(surf, work / "views.bin");
  save(work / "photo-gains.bin", gains);
  ifstream f(work / "views.bin", ios::binary);
  int count;
  read(f, &count, 1);
  vector<array<float, 4>> blend(S * S);
  vector<array<float, 4>> broad(S * S);
  vector<float> peak(S * S);
  vector<Vec3> xyz(S * S);
  vector<uint8_t> mask(S * S);
  View view;
  for (int i = 0; i < count; i++) {
    view.load(f);
    view.measureFocus();
#pragma omp parallel for schedule(dynamic, 8)
    for (size_t j = 0; j < surf.bins.size(); j++) {
      auto& b = surf.bins[j];
      if (!view.visible(b.center)) continue;
      for (const auto& s : b.samples) {
        Vec3 c;
        float quality = 0;
        float support = view.sample(s, c, &quality);
        if (support <= 0) continue;
        support *= support;
        // A peaked but continuous blend keeps one or two well-registered source
        // photos dominant. Normalize to the running maximum to retain weakly seen areas.
        auto& best = peak[s.pixel];
        auto& out = blend[s.pixel];
        auto power8 = [](float x) {
          float x2 = x * x, x4 = x2 * x2;
          return x4 * x4;
        };
        if (quality > best) {
          float factor = power8(best / quality);
          for (float& value : out) value *= factor;
          best = quality;
        }
        float w = power8(quality / best);
        auto& average = broad[s.pixel];
        for (int k = 0; k < 3; k++) {
          float sample = min(255.f, c[k] * gains[i]);
          out[k] += w * sample;
          average[k] += support * sample;
        }
        average[3] += support;
        out[3] += w;
      }
    }
  }
  vector<uint8_t> rgb(S * S * 3);
  for (auto& b : surf.bins)
    for (const auto& s : b.samples) {
      auto& c = blend[s.pixel];
      mask[s.pixel] = c[3] > 0 ? 2 : 1;
      xyz[s.pixel] = s.p;
      if (c[3] > 0)
        for (int k = 0; k < 3; k++) {
          const auto& average = broad[s.pixel];
          // Retain the broader exposure estimate around highlights and slight misregistrations.
          float sample = .67f * c[k] / c[3] + .33f * average[k] / average[3];
          rgb[3 * s.pixel + k] = clamp(lround(255 * pow(sample / 255, .82f)), 0L, 255L);
        }
    }
  save(work / "atlas.bgr", rgb);
  save(work / "world.bin", xyz);
  save(work / "mask.bin", mask);
  cout << count << " photos, " << std::count(mask.begin(), mask.end(), 2) << " observed texels / "
       << std::count_if(mask.begin(), mask.end(), [](int f) { return f != 0; }) << endl;
}
