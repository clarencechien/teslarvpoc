// Composite camera-aligned Gaussians back-to-front onto inverse-depth planes.
// Input coordinates are millimetres in a right/up/back camera frame.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>

struct Splat { int16_t x,y,z; uint16_t sigma; uint8_t r,g,b,a; };
static_assert(sizeof(Splat)==12);
constexpr int count=32, width=768, height=576;
constexpr float cx=383.5f, cy=287.5f, focal=525.f;
int main(int argc,char** argv){
    if(argc!=3)return 1;
    std::ifstream in(argv[1],std::ios::binary);
    if(!in)return 2;
    std::array<std::vector<Splat>,count> layers;
    Splat s;
    while(in.read(reinterpret_cast<char*>(&s),sizeof(s))){
        float disparity=std::clamp(1000.f/(-s.z),.14f,2.9f);
        int bin=std::lround((disparity-.14f)*(count-1)/(2.9f-.14f));
        layers[bin].push_back(s);
    }
    for(int i=0;i<count;i++){
        if(layers[i].empty())continue;
        std::vector<std::array<float,4>> pixels(width*height);
        for(const Splat& p:layers[i]){
            float depth=-p.z*.001f, sx=p.x*.001f/depth, sy=p.y*.001f/depth;
            float u=cx+focal*sx, v=cy-focal*sy, size=focal*p.sigma*.000001f/depth;
            float a=size*size*(1+.09f*sx*sx)+.08f;
            float b=-size*size*.09f*sx*sy;
            float c=size*size*(1+.09f*sy*sy)+.08f;
            float det=a*c-b*b, half=3*std::sqrt(std::max(a,c));
            for(int y=std::max(0,int(std::floor(v-half)));y<=std::min(height-1,int(std::ceil(v+half)));y++){
                for(int x=std::max(0,int(std::floor(u-half)));x<=std::min(width-1,int(std::ceil(u+half)));x++){
                    float dx=x-u,dy=y-v;
                    float power=-.5f*(c*dx*dx-2*b*dx*dy+a*dy*dy)/det;
                    if(power < -4.2f)continue;
                    float alpha=std::exp(power)*p.a/255.f;
                    auto& dst=pixels[y*width+x];
                    std::array<float,4> src={float(p.r),float(p.g),float(p.b),255.f};
                    for(int k=0;k<4;k++)dst[k]=src[k]*alpha+dst[k]*(1-alpha);
                }
            }
        }
        std::vector<uint8_t> packed(pixels.size()*4);
        for(size_t j=0;j<pixels.size();j++){
            const auto& v=pixels[j];
            if(v[3]<1)continue;
            for(int k=0;k<3;k++)packed[4*j+k]=std::clamp(std::lround(v[k]*255.f/v[3]),0L,255L);
            packed[4*j+3]=std::clamp(std::lround(v[3]),0L,255L);
        }
        std::ofstream file(std::string(argv[2])+"/"+std::to_string(i)+".rgba",std::ios::binary);
        if(!file.write(reinterpret_cast<const char*>(packed.data()),packed.size()))return 3;
    }
}
