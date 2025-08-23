// flip_fluid_opengl.cpp
// Single-file C++/OpenGL port skeleton of your WebGL FLIP demo.
// Focus: identical rendering (particles + obstacle disk), input, timing, and a
// scaffolding FlipFluid class mirroring your JS API so you can fill in
// the full solver step-by-step.
//
// ✅ Builds on Windows/Linux/macOS with:
//   - OpenGL 3.3 core, GLFW, GLAD
//   - (optional) GLM for math (we keep it minimal)
//
// Build (Linux/macOS, Homebrew-installed glfw):
//   g++ flip_fluid_opengl.cpp -std=c++20 -lglfw -ldl -lpthread -framework OpenGL -o flip_fluid
//   (on Linux without framework: add -lGL)
//   You may need to generate GLAD headers or switch to GLEW if you prefer.
//
// NOTE: For convenience, this file embeds a minimal GLAD loader (static) below.
// If you already have GLAD in your project, delete the embedded loader section
// and include <glad/glad.h> instead.
//
// --------------------------------------------------------------
// What works now:
//   - Window + OpenGL init (GLFW + GLAD)
//   - Shaders equivalent to your JS versions
//   - Particle and grid buffers
//   - Red obstacle disk you can drag with mouse/touch-like input
//   - Start/stop with P (pause toggle)
//   - Step one frame with M (single step)
//   - A lightweight particle-only integrator so you can see motion now
//
// What remains/TODOs (clearly marked):
//   - Full FLIP grid transfer, Poisson pressure solve, drift compensation, etc.
//   - Replacing the simple particle integrator with the full pipeline
//     mirrored from your JS code (the class layout already matches)
//
// This way you have a compiling, interactive OpenGL app that matches the
// structure and can accept your FLIP internals with minimal port friction.
// --------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <array>
#include <chrono>
#include <algorithm>



#include <glad/glad.h>
#include <GLFW/glfw3.h>

// Dear ImGui
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"


// ----------------------------- Shaders -----------------------------

static const char* POINT_VS = R"GLSL(
#version 330 core
layout(location=0) in vec2 attrPosition;
layout(location=1) in vec3 attrColor;
uniform vec2 domainSize;
uniform float pointSize;
uniform float drawDisk;
out vec3 fragColor;
out float fragDrawDisk;
void main(){
    vec2 st = vec2(2.0/domainSize.x, 2.0/domainSize.y);
    vec2 off = vec2(-1.0, -1.0);
    gl_Position = vec4(attrPosition*st + off, 0.0, 1.0);
    gl_PointSize = pointSize;
    fragColor = attrColor;
    fragDrawDisk = drawDisk;
}
)GLSL";

static const char* POINT_FS = R"GLSL(
#version 330 core
in vec3 fragColor;
in float fragDrawDisk;
out vec4 outColor;
void main(){
    if (fragDrawDisk == 1.0) {
        vec2 c = gl_PointCoord - vec2(0.5);
        if (dot(c,c) > 0.25) discard;
    }
    outColor = vec4(fragColor, 1.0);
}
)GLSL";

static const char* MESH_VS = R"GLSL(
#version 330 core
layout(location=0) in vec2 attrPosition;
uniform vec2 domainSize;
uniform vec3 color;
uniform vec2 translation;
uniform float scale;
out vec3 fragColor;
void main(){
    vec2 v = translation + attrPosition * scale;
    vec2 st = vec2(2.0/domainSize.x, 2.0/domainSize.y);
    vec2 off = vec2(-1.0, -1.0);
    gl_Position = vec4(v*st + off, 0.0, 1.0);
    fragColor = color;
}
)GLSL";

static const char* MESH_FS = R"GLSL(
#version 330 core
in vec3 fragColor;
out vec4 outColor;
void main(){ outColor = vec4(fragColor,1.0); }
)GLSL";

// ----------------------- Utility helpers -----------------------

static GLuint compileShader(GLenum type, const char* src){
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok=0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if(!ok){
        GLint len=0; glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
        std::string log(len,'\0'); glGetShaderInfoLog(s,len,nullptr,log.data());
        fprintf(stderr,"Shader compile error:\n%s\n", log.c_str());
        exit(1);
    }
    return s;
}

static GLuint linkProgram(const char* vs, const char* fs){
    GLuint v = compileShader(GL_VERTEX_SHADER, vs);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, v); glAttachShader(p, f);
    glLinkProgram(p);
    GLint ok=0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if(!ok){
        GLint len=0; glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len);
        std::string log(len,'\0'); glGetProgramInfoLog(p,len,nullptr,log.data());
        fprintf(stderr,"Program link error:\n%s\n", log.c_str());
        exit(1);
    }
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

static float clampf(float x, float a, float b){ return std::max(a, std::min(b, x)); }

// -------------------------- FlipFluid --------------------------

struct FlipFluid {
    // Grid (same naming as JS)
    int fNumX{}, fNumY{}, fNumCells{}; // grid dims
    float h{};                         // cell size
    float fInvSpacing{};               // 1/h
    float density{};

    // Fields
    std::vector<float> u, v, du, dv, prevU, prevV, p, s; // scalar fields
    std::vector<int>   cellType;                         // 0=fluid,1=air,2=solid
    std::vector<float> cellColor;                        // rgb per cell

    // Particles
    int maxParticles{}; int numParticles{};
    float particleRadius{}; float pInvSpacing{};
    int pNumX{}, pNumY{}, pNumCells{};

    std::vector<float> particlePos;   // 2*numParticles
    std::vector<float> particleVel;   // 2*numParticles
    std::vector<float> particleColor; // 3*numParticles
    std::vector<float> particleDensity; // fNumCells
    float particleRestDensity{0.f};

    // Cell particle bins (for neighborhood)
    std::vector<int> numCellParticles;
    std::vector<int> firstCellParticle; // +1 guard
    std::vector<int> cellParticleIds;   // indices of particles

    FlipFluid(float density_, float width, float height, float spacing, float particleRadius_, int maxParticles_){
        density = density_;
        fNumX = int(std::floor(width/spacing))+1;
        fNumY = int(std::floor(height/spacing))+1;
        h = std::max(width/float(fNumX), height/float(fNumY));
        fInvSpacing = 1.f/h;
        fNumCells = fNumX * fNumY;

        u.assign(fNumCells,0); v.assign(fNumCells,0);
        du.assign(fNumCells,0); dv.assign(fNumCells,0);
        prevU.assign(fNumCells,0); prevV.assign(fNumCells,0);
        p.assign(fNumCells,0); s.assign(fNumCells,0);
        cellType.assign(fNumCells,0);
        cellColor.assign(3*fNumCells,0);

        maxParticles = maxParticles_;
        particlePos.assign(2*maxParticles,0);
        particleVel.assign(2*maxParticles,0);
        particleColor.assign(3*maxParticles,0);
        for(int i=0;i<maxParticles;i++) particleColor[3*i+2]=1.0f;
        particleDensity.assign(fNumCells,0);

        particleRadius = particleRadius_;
        pInvSpacing = 1.f/(2.2f*particleRadius);
        pNumX = int(std::floor(width * pInvSpacing))+1;
        pNumY = int(std::floor(height* pInvSpacing))+1;
        pNumCells = pNumX * pNumY;

        numCellParticles.assign(pNumCells,0);
        firstCellParticle.assign(pNumCells+1,0);
        cellParticleIds.assign(maxParticles,0);

        numParticles = 0;
    }

    void integrateParticles(float dt, float gravity){
        for(int i=0;i<numParticles;i++){
            particleVel[2*i+1] += dt * gravity;
            particlePos[2*i  ] += particleVel[2*i  ]*dt;
            particlePos[2*i+1] += particleVel[2*i+1]*dt;
        }
    }

    void handleParticleCollisions(float obstacleX, float obstacleY, float obstacleRadius, float obstacleVelX, float obstacleVelY){
        float r = particleRadius;
        float h_ = 1.0f / fInvSpacing;
        float minX = h_ + r, maxX = (fNumX-1)*h_ - r;
        float minY = h_ + r, maxY = (fNumY-1)*h_ - r;
        float minDist = obstacleRadius + r; float minDist2 = minDist*minDist;
        float wallBounce = -0.5f; // Damping factor for wall bounce
        for(int i=0;i<numParticles;i++){
            float& x = particlePos[2*i];
            float& y = particlePos[2*i+1];
            float& vx = particleVel[2*i];
            float& vy = particleVel[2*i+1];
            float dx = x - obstacleX, dy = y - obstacleY;
            float d2 = dx*dx + dy*dy;
            // Obstacle collision: set velocity to obstacle's velocity if inside
            if(d2 < minDist2){
                vx = obstacleVelX;
                vy = obstacleVelY;
            }
            // Wall collisions (reflect and damp velocity)
            if(x < minX){ x=minX; vx *= wallBounce; }
            if(x > maxX){ x=maxX; vx *= wallBounce; }
            if(y < minY){ y=minY; vy *= wallBounce; }
            if(y > maxY){ y=maxY; vy *= wallBounce; }
        }
        // Debug: print particles near the bottom wall after collision handling
        int count = 0;
        for(int i=0;i<numParticles;i++){
            float y = particlePos[2*i+1];
            if(y < minY + 2.0f * r){
                if(count < 10) // limit output
                    printf("[DEBUG] After collision: Particle %d y=%.4f vy=%.4f\n", i, y, particleVel[2*i+1]);
                count++;
            }
        }
        if(count > 0) printf("[DEBUG] Total particles near bottom wall after collision: %d\n", count);
    }

    // Particle separation (limit max displacement per step for stability)
    void pushParticlesApart(int numIters) {
        float minDist = 2.0f * particleRadius;
        float minDist2 = minDist * minDist;
        float colorDiffusionCoeff = 0.001f;
        for (int iter = 0; iter < numIters; ++iter) {
            // --- Spatial binning (column-major) ---
            std::fill(numCellParticles.begin(), numCellParticles.end(), 0);
            for (int i = 0; i < numParticles; ++i) {
                float x = particlePos[2*i];
                float y = particlePos[2*i+1];
                int xi = std::clamp(int(x * pInvSpacing), 0, pNumX-1);
                int yi = std::clamp(int(y * pInvSpacing), 0, pNumY-1);
                int cellNr = xi * pNumY + yi;
                numCellParticles[cellNr]++;
            }
            // Compute prefix sum for firstCellParticle
            firstCellParticle[0] = 0;
            for (int i = 0; i < pNumCells; ++i) {
                firstCellParticle[i+1] = firstCellParticle[i] + numCellParticles[i];
                numCellParticles[i] = 0; // reset for use as counter below
            }
            // Fill cellParticleIds
            for (int i = 0; i < numParticles; ++i) {
                float x = particlePos[2*i];
                float y = particlePos[2*i+1];
                int xi = std::clamp(int(x * pInvSpacing), 0, pNumX-1);
                int yi = std::clamp(int(y * pInvSpacing), 0, pNumY-1);
                int cellNr = xi * pNumY + yi;
                int idx = firstCellParticle[cellNr] + numCellParticles[cellNr];
                cellParticleIds[idx] = i;
                numCellParticles[cellNr]++;
            }
            // --- Push apart using neighbors in bins ---
            for (int i = 0; i < numParticles; ++i) {
                float px = particlePos[2*i];
                float py = particlePos[2*i+1];
                int pxi = std::clamp(int(px * pInvSpacing), 0, pNumX-1);
                int pyi = std::clamp(int(py * pInvSpacing), 0, pNumY-1);
                for (int dx = -1; dx <= 1; ++dx) {
                    int xi = pxi + dx;
                    if (xi < 0 || xi >= pNumX) continue;
                    for (int dy = -1; dy <= 1; ++dy) {
                        int yi = pyi + dy;
                        if (yi < 0 || yi >= pNumY) continue;
                        int cellNr = xi * pNumY + yi;
                        int first = firstCellParticle[cellNr];
                        int last = firstCellParticle[cellNr+1];
                        for (int jidx = first; jidx < last; ++jidx) {
                            int j = cellParticleIds[jidx];
                            if (j <= i) continue; // avoid double-pushing same pair
                            float qx = particlePos[2*j];
                            float qy = particlePos[2*j+1];
                            float dx = qx - px;
                            float dy = qy - py;
                            float d2 = dx*dx + dy*dy;
                            if (d2 > minDist2 || d2 == 0.0f) continue;
                            float d = std::sqrt(d2);
                            float s = 0.5f * (minDist - d) / d;
                            float pushx = dx * s;
                            float pushy = dy * s;
                            particlePos[2*i] -= pushx;
                            particlePos[2*i+1] -= pushy;
                            particlePos[2*j] += pushx;
                            particlePos[2*j+1] += pushy;
                            // Color diffusion
                            for (int k = 0; k < 3; ++k) {
                                float color0 = particleColor[3*i+k];
                                float color1 = particleColor[3*j+k];
                                float color = 0.5f * (color0 + color1);
                                particleColor[3*i+k] = color0 + (color - color0) * colorDiffusionCoeff;
                                particleColor[3*j+k] = color1 + (color - color1) * colorDiffusionCoeff;
                            }
                        }
                    }
                }
            }
        }
    }

    // Update density at each grid cell by counting nearby particles
    void updateParticleDensity() {
        std::fill(particleDensity.begin(), particleDensity.end(), 0.0f);
        float h2 = 0.5f * h;
        float h1 = fInvSpacing;
        int n = fNumY;
        for (int i = 0; i < numParticles; ++i) {
            float x = clampf(particlePos[2*i], h, (fNumX-1)*h);
            float y = clampf(particlePos[2*i+1], h, (fNumY-1)*h);

            int x0 = int(std::floor((x - h2) * h1));
            float tx = ((x - h2) - x0 * h) * h1;
            int x1 = std::min(x0 + 1, fNumX-2);

            int y0 = int(std::floor((y - h2) * h1));
            float ty = ((y - h2) - y0 * h) * h1;
            int y1 = std::min(y0 + 1, fNumY-2);

            float sx = 1.0f - tx;
            float sy = 1.0f - ty;

            if (x0 >= 0 && x0 < fNumX && y0 >= 0 && y0 < fNumY)
                particleDensity[x0 * n + y0] += sx * sy;
            if (x1 >= 0 && x1 < fNumX && y0 >= 0 && y0 < fNumY)
                particleDensity[x1 * n + y0] += tx * sy;
            if (x1 >= 0 && x1 < fNumX && y1 >= 0 && y1 < fNumY)
                particleDensity[x1 * n + y1] += tx * ty;
            if (x0 >= 0 && x0 < fNumX && y1 >= 0 && y1 < fNumY)
                particleDensity[x0 * n + y1] += sx * ty;
        }

        // Compute rest density once, as in JS
        if (particleRestDensity == 0.0f) {
            float sum = 0.0f;
            int numFluidCells = 0;
            for (int i = 0; i < fNumCells; i++) {
                if (cellType[i] == 0) { // FLUID_CELL
                    sum += particleDensity[i];
                    numFluidCells++;
                }
            }
            if (numFluidCells > 0)
                particleRestDensity = sum / numFluidCells;
        }
    }

    // FLIP/PIC velocity transfer
    void transferVelocities(bool toGrid, float flipRatio=0.0f, const std::vector<float>* oldParticleVelPtr=nullptr) {
        if (toGrid) {
            std::fill(du.begin(), du.end(), 0.f);
            std::fill(dv.begin(), dv.end(), 0.f);
            std::fill(u.begin(), u.end(), 0.f);
            std::fill(v.begin(), v.end(), 0.f);
            std::vector<float> weightU(fNumCells, 0.0f), weightV(fNumCells, 0.0f);

            // --- JS-style: Reset cellType to AIR or SOLID, then mark FLUID cells with particles ---
            for (int i = 0; i < fNumX; ++i) {
                for (int j = 0; j < fNumY; ++j) {
                    int idx = i * fNumY + j;
                    if (s[idx] == 0.0f) cellType[idx] = 2; // SOLID
                    else cellType[idx] = 1; // AIR
                }
            }
            for (int p = 0; p < numParticles; ++p) {
                float x = particlePos[2*p];
                float y = particlePos[2*p+1];
                int gx = int(x * fInvSpacing);
                int gy = int(y * fInvSpacing);
                if (gx >= 0 && gx < fNumX && gy >= 0 && gy < fNumY) {
                    int idx = gx * fNumY + gy;
                    if (cellType[idx] == 1) cellType[idx] = 0; // FLUID
                }
            }

            // --- Bilinear splat with MAC offsets ---
            // U component (x-velocity) at (i+0.5, j)
            for (int p = 0; p < numParticles; ++p) {
                float x = clampf(particlePos[2*p], h, (fNumX-1)*h);
                float y = clampf(particlePos[2*p+1], h, (fNumY-1)*h);
                float vx = particleVel[2*p];
                // MAC offset: sample at (x, y+0.5h)
                float px = x;
                float py = y + 0.5f * h;
                float fx = (px - h) / h;
                float fy = (py - h) / h;
                int ix = std::min(std::max(int(fx), 0), fNumX-2);
                int iy = std::min(std::max(int(fy), 0), fNumY-2);
                float tx = fx - ix;
                float ty = fy - iy;
                float w00 = (1-tx)*(1-ty);
                float w10 = tx*(1-ty);
                float w11 = tx*ty;
                float w01 = (1-tx)*ty;
                int idx00 = ix*fNumY + iy;
                int idx10 = (ix+1)*fNumY + iy;
                int idx11 = (ix+1)*fNumY + (iy+1);
                int idx01 = ix*fNumY + (iy+1);
                if (cellType[idx00]==0) { u[idx00] += vx*w00; weightU[idx00] += w00; }
                if (cellType[idx10]==0) { u[idx10] += vx*w10; weightU[idx10] += w10; }
                if (cellType[idx11]==0) { u[idx11] += vx*w11; weightU[idx11] += w11; }
                if (cellType[idx01]==0) { u[idx01] += vx*w01; weightU[idx01] += w01; }
            }
            // V component (y-velocity) at (i, j+0.5)
            for (int p = 0; p < numParticles; ++p) {
                float x = clampf(particlePos[2*p], h, (fNumX-1)*h);
                float y = clampf(particlePos[2*p+1], h, (fNumY-1)*h);
                float vy = particleVel[2*p+1];
                // MAC offset: sample at (x+0.5h, y)
                float px = x + 0.5f * h;
                float py = y;
                float fx = (px - h) / h;
                float fy = (py - h) / h;
                int ix = std::min(std::max(int(fx), 0), fNumX-2);
                int iy = std::min(std::max(int(fy), 0), fNumY-2);
                float tx = fx - ix;
                float ty = fy - iy;
                float w00 = (1-tx)*(1-ty);
                float w10 = tx*(1-ty);
                float w11 = tx*ty;
                float w01 = (1-tx)*ty;
                int idx00 = ix*fNumY + iy;
                int idx10 = (ix+1)*fNumY + iy;
                int idx11 = (ix+1)*fNumY + (iy+1);
                int idx01 = ix*fNumY + (iy+1);
                if (cellType[idx00]==0) { v[idx00] += vy*w00; weightV[idx00] += w00; }
                if (cellType[idx10]==0) { v[idx10] += vy*w10; weightV[idx10] += w10; }
                if (cellType[idx11]==0) { v[idx11] += vy*w11; weightV[idx11] += w11; }
                if (cellType[idx01]==0) { v[idx01] += vy*w01; weightV[idx01] += w01; }
            }
            for (int i = 0; i < fNumCells; ++i) {
                if (weightU[i] > 0) u[i] /= weightU[i];
                if (weightV[i] > 0) v[i] /= weightV[i];
            }
            // Restore solid cell velocities (and neighbors) to previous values
            for (int i = 0; i < fNumX; ++i) {
                for (int j = 0; j < fNumY; ++j) {
                    int idx = i*fNumY + j;
                    if (cellType[idx]==2 || (i>0 && cellType[(i-1)*fNumY+j]==2))
                        u[idx] = prevU[idx];
                    if (cellType[idx]==2 || (j>0 && cellType[i*fNumY+(j-1)]==2))
                        v[idx] = prevV[idx];
                }
            }
        } else {
            // Debug: Confirm FLIP branch is running and print some velocities
            printf("[DEBUG] FLIP branch entered. flipRatio=%.3f\n", flipRatio);
            for (int p = 0; p < std::min(5, numParticles); ++p) {
                float vx = particleVel[2*p];
                float vy = particleVel[2*p+1];
                printf("[DEBUG] Before FLIP: Particle %d vx=%.4f vy=%.4f\n", p, vx, vy);
            }
            // Grid to particle (PIC/FLIP blend)
            float maxVel = 10.0f; // clamp velocity magnitude
            const std::vector<float>& baseParticleVel = oldParticleVelPtr ? *oldParticleVelPtr : particleVel;
            for (int p = 0; p < numParticles; ++p) {
                float x = clampf(particlePos[2*p], h, (fNumX-1)*h);
                float y = clampf(particlePos[2*p+1], h, (fNumY-1)*h);
                // U component (x-velocity) at (x, y+0.5h)
                float px_u = x;
                float py_u = y + 0.5f * h;
                float fx_u = (px_u - h) / h;
                float fy_u = (py_u - h) / h;
                int ix_u = std::min(std::max(int(fx_u), 0), fNumX-2);
                int iy_u = std::min(std::max(int(fy_u), 0), fNumY-2);
                float tx_u = fx_u - ix_u;
                float ty_u = fy_u - iy_u;
                float w00_u = (1-tx_u)*(1-ty_u);
                float w10_u = tx_u*(1-ty_u);
                float w11_u = tx_u*ty_u;
                float w01_u = (1-tx_u)*ty_u;
                int idx00_u = ix_u*fNumY + iy_u;
                int idx10_u = (ix_u+1)*fNumY + iy_u;
                int idx11_u = (ix_u+1)*fNumY + (iy_u+1);
                int idx01_u = ix_u*fNumY + (iy_u+1);
                float picU = 0, prevPicU = 0, wsumU = 0;
                if (cellType[idx00_u]==0) { picU += u[idx00_u]*w00_u; prevPicU += prevU[idx00_u]*w00_u; wsumU += w00_u; }
                if (cellType[idx10_u]==0) { picU += u[idx10_u]*w10_u; prevPicU += prevU[idx10_u]*w10_u; wsumU += w10_u; }
                if (cellType[idx11_u]==0) { picU += u[idx11_u]*w11_u; prevPicU += prevU[idx11_u]*w11_u; wsumU += w11_u; }
                if (cellType[idx01_u]==0) { picU += u[idx01_u]*w01_u; prevPicU += prevU[idx01_u]*w01_u; wsumU += w01_u; }
                float baseU = baseParticleVel[2*p];
                if (wsumU > 0) {
                    picU /= wsumU; prevPicU /= wsumU;
                } else {
                    picU = baseU; prevPicU = baseU;
                }

                // V component (y-velocity) at (x+0.5h, y)
                float px_v = x + 0.5f * h;
                float py_v = y;
                float fx_v = (px_v - h) / h;
                float fy_v = (py_v - h) / h;
                int ix_v = std::min(std::max(int(fx_v), 0), fNumX-2);
                int iy_v = std::min(std::max(int(fy_v), 0), fNumY-2);
                float tx_v = fx_v - ix_v;
                float ty_v = fy_v - iy_v;
                float w00_v = (1-tx_v)*(1-ty_v);
                float w10_v = tx_v*(1-ty_v);
                float w11_v = tx_v*ty_v;
                float w01_v = (1-tx_v)*ty_v;
                int idx00_v = ix_v*fNumY + iy_v;
                int idx10_v = (ix_v+1)*fNumY + iy_v;
                int idx11_v = (ix_v+1)*fNumY + (iy_v+1);
                int idx01_v = ix_v*fNumY + (iy_v+1);
                float picV = 0, prevPicV = 0, wsumV = 0;
                if (cellType[idx00_v]==0) { picV += v[idx00_v]*w00_v; prevPicV += prevV[idx00_v]*w00_v; wsumV += w00_v; }
                if (cellType[idx10_v]==0) { picV += v[idx10_v]*w10_v; prevPicV += prevV[idx10_v]*w10_v; wsumV += w10_v; }
                if (cellType[idx11_v]==0) { picV += v[idx11_v]*w11_v; prevPicV += prevV[idx11_v]*w11_v; wsumV += w11_v; }
                if (cellType[idx01_v]==0) { picV += v[idx01_v]*w01_v; prevPicV += prevV[idx01_v]*w01_v; wsumV += w01_v; }
                float baseV = baseParticleVel[2*p+1];
                if (wsumV > 0) {
                    picV /= wsumV; prevPicV /= wsumV;
                } else {
                    picV = baseV; prevPicV = baseV;
                }

                float flipU = baseU + (picU - prevPicU);
                float flipV = baseV + (picV - prevPicV);
                float newU = (1.0f - flipRatio) * picU + flipRatio * flipU;
                float newV = (1.0f - flipRatio) * picV + flipRatio * flipV;
                // Clamp velocity magnitude
                float mag = std::sqrt(newU*newU + newV*newV);
                if (mag > maxVel) {
                    newU *= maxVel / mag;
                    newV *= maxVel / mag;
                }
                particleVel[2*p] = newU;
                particleVel[2*p+1] = newV;
                if (p < 5) {
                    printf("[DEBUG] After FLIP:  Particle %d vx=%.4f vy=%.4f\n", p, newU, newV);
                }
            }
        }
    }

    void solveIncompressibility(int numIters, float dt, float overRelaxation, bool compensateDrift) {
        // JS-style SOR pressure solve: update both pressure and velocities in-place using s weights
        int nX = fNumX, nY = fNumY;
        std::fill(p.begin(), p.end(), 0.0f);
        std::vector<float> prevU = u;
        std::vector<float> prevV = v;
        float cp = density * h / dt;

        for (int iter = 0; iter < numIters; ++iter) {
            for (int i = 1; i < nX - 1; ++i) {
                for (int j = 1; j < nY - 1; ++j) {
                    int idx = i * nY + j;
                    if (cellType[idx] != 0) continue; // Only fluid

                    int left   = (i-1)*nY + j;
                    int right  = (i+1)*nY + j;
                    int bottom = i*nY + (j-1);
                    int top    = i*nY + (j+1);

                    float sx0 = s[left];
                    float sx1 = s[right];
                    float sy0 = s[bottom];
                    float sy1 = s[top];
                    float sumS = sx0 + sx1 + sy0 + sy1;
                    if (sumS == 0.0f) continue;

                    float div = u[right] - u[idx] + v[top] - v[idx];
                    // Drift compensation (as in JS)
                    if (compensateDrift && particleRestDensity > 0.0f) {
                        float compression = particleDensity[idx] - particleRestDensity;
                        if (compression > 0.0f) {
                            float k = 1.0f;
                            div -= k * compression;
                        }
                    }

                    float pCorr = -div / sumS;
                    pCorr *= overRelaxation;
                    p[idx] += cp * pCorr;

                    u[idx]     -= sx0 * pCorr;
                    u[right]   += sx1 * pCorr;
                    v[idx]     -= sy0 * pCorr;
                    v[top]     += sy1 * pCorr;
                }
            }
        }
    }

    void updateParticleColors(){
        // Color based on speed: blue (slow) to white (fast)
        float maxSpeed = 5.0f;
        for(int i=0;i<numParticles;i++){
            float vx = particleVel[2*i];
            float vy = particleVel[2*i+1];
            float speed = std::sqrt(vx*vx + vy*vy);
            float t = clampf(speed / maxSpeed, 0.0f, 1.0f);
            // Base blue color
            float baseR = 0.2f, baseG = 0.4f, baseB = 1.0f;
            // Blend with white as speed increases
            particleColor[3*i+0] = baseR * (1.0f - t) + 1.0f * t;
            particleColor[3*i+1] = baseG * (1.0f - t) + 1.0f * t;
            particleColor[3*i+2] = baseB * (1.0f - t) + 1.0f * t;
        }
    }

    void updateCellColors(){
        // Gray for solids, blue-to-white for fluid based on velocity magnitude
        for(int i=0;i<fNumCells;i++){
            float r=0,g=0,b=0;
            if(cellType[i]==2){ r=g=b=0.5f; }
            else if(cellType[i]==0){ // fluid
                // Compute velocity magnitude at this cell
                float vel = std::sqrt(u[i]*u[i] + v[i]*v[i]);
                // Map velocity to [0,1] for blending (tune maxVel as needed)
                float maxVel = 5.0f;
                float t = clampf(vel / maxVel, 0.0f, 1.0f);
                // Base blue color
                float baseR = 0.2f, baseG = 0.4f, baseB = 1.0f;
                // Blend with white as velocity increases
                r = baseR * (1.0f - t) + 1.0f * t;
                g = baseG * (1.0f - t) + 1.0f * t;
                b = baseB * (1.0f - t) + 1.0f * t;
            }
            cellColor[3*i+0]=r; cellColor[3*i+1]=g; cellColor[3*i+2]=b;
        }
    }

    void simulate(float dt, float gravity, float flipRatio, int numPressureIters,
                  int numParticleIters, float overRelaxation, bool compensateDrift,
                  bool separateParticles, float obstacleX, float obstacleY, float obstacleRadius,
                  float obstacleVelX, float obstacleVelY){
        // Substeps can be 1 for now
        integrateParticles(dt, gravity);
        if(separateParticles) pushParticlesApart(numParticleIters);
        handleParticleCollisions(obstacleX, obstacleY, obstacleRadius, obstacleVelX, obstacleVelY);
        // Store a copy of particleVel before grid transfer for FLIP
        std::vector<float> oldParticleVel = particleVel;
        prevU = u; prevV = v;
        transferVelocities(true);
        // (No need to apply gravity to grid velocities here; handled by particles)
        updateParticleDensity();
        solveIncompressibility(numPressureIters, dt, overRelaxation, compensateDrift);
        // Zero grid velocities at solid boundaries (post-pressure solve)
        for (int i = 0; i < fNumX; ++i) {
            for (int j = 0; j < fNumY; ++j) {
                int idx = i * fNumY + j;
                if (cellType[idx] == 2) { // solid
                    u[idx] = 0.0f;
                    v[idx] = 0.0f;
                }
            }
        }
        transferVelocities(false, flipRatio, &oldParticleVel);
        // Debug: print particles near the bottom wall after pressure solve and velocity transfer
        float h_ = 1.0f / fInvSpacing;
        float minY = h_ + particleRadius;
        int count = 0;
        for(int i=0;i<numParticles;i++){
            float y = particlePos[2*i+1];
            if(y < minY + 2.0f * particleRadius){
                if(count < 10)
                    printf("[DEBUG] After pressure solve: Particle %d y=%.4f vy=%.4f\n", i, y, particleVel[2*i+1]);
                count++;
            }
        }
        if(count > 0) printf("[DEBUG] Total particles near bottom wall after pressure solve: %d\n", count);
        updateParticleColors();
        updateCellColors();
    }
};

// --------------------------- Scene state ---------------------------

struct Scene {
    float gravity = -9.81f;
    float dt = 1.f / 60.f;
    float flipRatio = 0.f;
    int   numPressureIters = 25;
    int   numParticleIters = 2;
    int   frameNr = 0;
    float overRelaxation = 1.9f;
    bool  compensateDrift = true;
    bool  separateParticles = true;
    float obstacleX = 0.f, obstacleY = 0.f, obstacleRadius = 0.15f;
    bool  paused = false;
    bool  showObstacle = true;
    float obstacleVelX = 0.f, obstacleVelY = 0.f;
    bool  showParticles = true;
    bool  showGrid = false;
    FlipFluid* fluid = nullptr;

    float simWidth=0.f, simHeight=0.f;
} scene;

// ---------------------------- Geometry ----------------------------


struct GLObjects {
    GLuint pointProg=0, meshProg=0;
    GLuint gridVBO=0, gridColorVBO=0;
    GLuint particleVBO=0, particleColorVBO=0;
    GLuint diskVBO=0, diskEBO=0;
    GLuint globalVAO=0;
} glb;

// --- Buffer creation helpers (mirroring JS logic) ---
void createOrUpdateGridVBO() {
    auto& f = *scene.fluid;
    if (glb.gridVBO == 0) glGenBuffers(1, &glb.gridVBO);
    if (glb.gridColorVBO == 0) glGenBuffers(1, &glb.gridColorVBO);
    std::vector<float> cellCenters(2 * f.fNumCells);
    int p = 0;
    for (int i = 0; i < f.fNumX; i++) {
        for (int j = 0; j < f.fNumY; j++) {
            cellCenters[p++] = (i + 0.5f) * f.h;
            cellCenters[p++] = (j + 0.5f) * f.h;
        }
    }
    glBindBuffer(GL_ARRAY_BUFFER, glb.gridVBO);
    glBufferData(GL_ARRAY_BUFFER, cellCenters.size() * sizeof(float), cellCenters.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void createParticleVBOs() {
    if (glb.particleVBO == 0) glGenBuffers(1, &glb.particleVBO);
    if (glb.particleColorVBO == 0) glGenBuffers(1, &glb.particleColorVBO);
}

void createDiskBuffers() {
    const int numSegs = 50;
    if (glb.diskVBO == 0) {
        glGenBuffers(1, &glb.diskVBO);
        std::vector<float> diskVerts(2 * (numSegs + 1));
        diskVerts[0] = 0.0f; diskVerts[1] = 0.0f;
        for (int i = 0; i < numSegs; i++) {
            float angle = 2.0f * float(M_PI) * float(i) / float(numSegs);
            diskVerts[2 * (i + 1) + 0] = std::cos(angle);
            diskVerts[2 * (i + 1) + 1] = std::sin(angle);
        }
        glBindBuffer(GL_ARRAY_BUFFER, glb.diskVBO);
        glBufferData(GL_ARRAY_BUFFER, diskVerts.size() * sizeof(float), diskVerts.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    if (glb.diskEBO == 0) {
        glGenBuffers(1, &glb.diskEBO);
        std::vector<unsigned short> diskIds(3 * numSegs);
        int p = 0;
        for (int i = 0; i < numSegs; i++) {
            diskIds[p++] = 0;
            diskIds[p++] = 1 + i;
            diskIds[p++] = 1 + ((i + 1) % numSegs);
        }
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glb.diskEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, diskIds.size() * sizeof(unsigned short), diskIds.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }
}

// --------------------------- Setup scene ---------------------------

void setObstacle(FlipFluid& f, float x, float y, bool reset){
    float vx=0, vy=0;
    if(!reset){ vx = (x - scene.obstacleX)/scene.dt; vy=(y-scene.obstacleY)/scene.dt; }
    scene.obstacleX = x; scene.obstacleY = y; scene.obstacleVelX=vx; scene.obstacleVelY=vy; scene.showObstacle=true;

    int n = f.fNumY;
    for(int i=1;i<f.fNumX-2;i++){
        for(int j=1;j<f.fNumY-2;j++){
            f.s[i*n+j] = 1.0f; // fluid by default
            float cx = (i+0.5f)*f.h, cy=(j+0.5f)*f.h;
            float dx = cx-x, dy=cy-y;
            if(dx*dx+dy*dy < scene.obstacleRadius*scene.obstacleRadius){
                f.s[i*n+j] = 0.0f; // solid
                f.u[i*n+j] = vx; f.u[(i+1)*n + j] = vx;
                f.v[i*n+j] = vy; f.v[i*n+j+1] = vy;
            }
        }
    }
}

void setupScene(float canvasWidth, float canvasHeight){
    float simHeight = 3.0f;
    float cScale = canvasHeight / simHeight; // pixels per unit (not used directly, but mirrors JS)
    float simWidth  = canvasWidth / cScale;
    scene.simWidth=simWidth; scene.simHeight=simHeight;

    float tankHeight = simHeight;
    float tankWidth  = simWidth;
    int res = 100; float h = tankHeight/float(res);
    float density = 1000.0f;

    float relWaterHeight = 0.8f;
    float relWaterWidth  = 0.6f;

    float r = 0.3f*h; float dx = 2.0f*r; float dy = std::sqrt(3.0f)/2.0f*dx;
    int numX = int(std::floor((relWaterWidth*tankWidth  - 2.0f*h - 2.0f*r)/dx));
    int numY = int(std::floor((relWaterHeight*tankHeight - 2.0f*h - 2.0f*r)/dy));
    int maxParticles = numX*numY;

    auto* f = new FlipFluid(density, tankWidth, tankHeight, h, r, maxParticles);
    scene.fluid = f;

    // Fill the tank with a block of blue fluid particles (hexagonal packing)
    int p = 0;
    for(int j=0;j<numY;j++){
        for(int i=0;i<numX;i++){
            if(p >= f->maxParticles) break;
            float x = h + r + i*dx + ((j&1) ? dx*0.5f : 0.0f);
            float y = h + r + j*dy;
            f->particlePos[2*p  ] = x;
            f->particlePos[2*p+1] = y;
            f->particleVel[2*p  ] = 0.0f;
            f->particleVel[2*p+1] = 0.0f;
            f->particleColor[3*p+0] = 0.2f; // blue
            f->particleColor[3*p+1] = 0.4f;
            f->particleColor[3*p+2] = 1.0f;
            p++;
        }
    }
    f->numParticles = p;

    // Tank boundaries and fluid/air assignment
    int n = f->fNumY;
    for(int i=0;i<f->fNumX;i++){
        for(int j=0;j<f->fNumY;j++){
            float ss = 1.0f; // fluid by default
            int type = 1; // air by default
            if(i==0 || i==f->fNumX-1 || j==0) {
                ss = 0.0f; // solid walls
                type = 2;
            } else {
                // If inside the initial water block, mark as fluid
                float cx = (i+0.5f)*f->h, cy = (j+0.5f)*f->h;
                if (cx > f->h && cx < f->h + relWaterWidth*tankWidth &&
                    cy > f->h && cy < f->h + relWaterHeight*tankHeight) {
                    type = 0; // fluid
                }
            }
            f->s[i*n + j] = ss;
            f->cellType[i*n + j] = type;
        }
    }

    // Ensure all grid cells containing a particle are marked as fluid
    for (int pi = 0; pi < f->numParticles; ++pi) {
        float x = f->particlePos[2*pi];
        float y = f->particlePos[2*pi+1];
        int gx = int(x * f->fInvSpacing);
        int gy = int(y * f->fInvSpacing);
        if (gx >= 0 && gx < f->fNumX && gy >= 0 && gy < f->fNumY) {
            int idx = gx * f->fNumY + gy;
            if (f->cellType[idx] != 2) // don't overwrite solids
                f->cellType[idx] = 0;
        }
    }

    setObstacle(*f, 3.0f, 2.0f, true);
}

// ----------------------------- Input -----------------------------

void draw(){
    int width = 0, height = 0;
    GLFWwindow* win = glfwGetCurrentContext();
    if (win) {
        glfwGetFramebufferSize(win, &width, &height);
        glViewport(0, 0, width, height);
    }
    glClearColor(0,0,0,1); glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

    // --- Bind global VAO for OpenGL core profile ---
    glBindVertexArray(glb.globalVAO);

    // --- Ensure OpenGL state for visible points ---
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);

    if(glb.pointProg==0) glb.pointProg = linkProgram(POINT_VS, POINT_FS);
    if(glb.meshProg==0)  glb.meshProg  = linkProgram(MESH_VS,  MESH_FS);

    // Grid (optional)
    if(glb.gridVBO==0) createOrUpdateGridVBO();
    if(scene.showGrid){
        auto& f = *scene.fluid;
        glUseProgram(glb.pointProg);
        GLint dom = glGetUniformLocation(glb.pointProg, "domainSize");
        GLint psize = glGetUniformLocation(glb.pointProg, "pointSize");
        GLint dd = glGetUniformLocation(glb.pointProg, "drawDisk");
        glUniform2f(dom, scene.simWidth, scene.simHeight);
        float pointSize = 0.9f * f.h / scene.simWidth * [](){int w,h;glfwGetFramebufferSize(glfwGetCurrentContext(),&w,&h);return (float)w;}();
        glUniform1f(psize, pointSize);
        glUniform1f(dd, 0.0f);

        glBindBuffer(GL_ARRAY_BUFFER, glb.gridVBO);
        glEnableVertexAttribArray(0); glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,0,(void*)0);

        glBindBuffer(GL_ARRAY_BUFFER, glb.gridColorVBO);
        auto bytes = scene.fluid->cellColor.size()*sizeof(float);
        glBufferData(GL_ARRAY_BUFFER, bytes, scene.fluid->cellColor.data(), GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(1); glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,0,(void*)0);

        glDrawArrays(GL_POINTS, 0, scene.fluid->fNumCells);

        glDisableVertexAttribArray(0); glDisableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER,0);
    }

    // Particles
    if(scene.showParticles){
        auto& f = *scene.fluid;
        glUseProgram(glb.pointProg);
        GLint dom = glGetUniformLocation(glb.pointProg, "domainSize");
        GLint psize = glGetUniformLocation(glb.pointProg, "pointSize");
        GLint dd = glGetUniformLocation(glb.pointProg, "drawDisk");
        glUniform2f(dom, scene.simWidth, scene.simHeight);
        float pointSize = 5.0f;
        glUniform1f(psize, pointSize);
        glUniform1f(dd, 1.0f);

        createParticleVBOs();
        glBindBuffer(GL_ARRAY_BUFFER, glb.particleVBO);
        glBufferData(GL_ARRAY_BUFFER, f.particlePos.size()*sizeof(float), f.particlePos.data(), GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(0); glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,0,(void*)0);

        glBindBuffer(GL_ARRAY_BUFFER, glb.particleColorVBO);
        glBufferData(GL_ARRAY_BUFFER, f.particleColor.size()*sizeof(float), f.particleColor.data(), GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(1); glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,0,(void*)0);

        glDrawArrays(GL_POINTS, 0, f.numParticles);

        // OpenGL error check
        GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            fprintf(stderr, "OpenGL error after particle draw: %x\n", err);
        }

        glDisableVertexAttribArray(0); glDisableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER,0);
    }

    // ...removed test green dot code...

    // Obstacle disk
    createDiskBuffers();
    glUseProgram(glb.meshProg);
    GLint dom = glGetUniformLocation(glb.meshProg, "domainSize");
    GLint col = glGetUniformLocation(glb.meshProg, "color");
    GLint tr  = glGetUniformLocation(glb.meshProg, "translation");
    GLint sc  = glGetUniformLocation(glb.meshProg, "scale");
    glUniform2f(dom, scene.simWidth, scene.simHeight);
    glUniform3f(col, 1.0f, 0.0f, 0.0f);
    glUniform2f(tr, scene.obstacleX, scene.obstacleY);
    glUniform1f(sc, scene.obstacleRadius + scene.fluid->particleRadius);

    glBindBuffer(GL_ARRAY_BUFFER, glb.diskVBO);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,0,(void*)0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glb.diskEBO);
    glDrawElements(GL_TRIANGLES, 3*50, GL_UNSIGNED_SHORT, 0);
    glDisableVertexAttribArray(0);
}




// ----------------------------- Input -----------------------------

static bool mouseDown=false;

void startDrag(double x, double y, GLFWwindow* win){
    int w,h; glfwGetFramebufferSize(win,&w,&h);
    // Convert pixels to sim units (origin at bottom-left, like JS conversion)
    float mx = (float)x; float my = (float)y;
    float cScale = h/scene.simHeight;
    float sx = mx / cScale; float sy = (h - my)/cScale;
    mouseDown = true;
    setObstacle(*scene.fluid, sx, sy, true);
    scene.paused = false;
}

void drag(double x, double y, GLFWwindow* win){
    if(!mouseDown) return;
    int w,h; glfwGetFramebufferSize(win,&w,&h);
    float mx = (float)x; float my = (float)y;
    float cScale = h/scene.simHeight;
    float sx = mx / cScale; float sy = (h - my)/cScale;
    setObstacle(*scene.fluid, sx, sy, false);
}

void endDrag(){ mouseDown=false; scene.obstacleVelX=0; scene.obstacleVelY=0; }


// ------------------------------ Main ------------------------------

int main(){
    if(!glfwInit()){ fprintf(stderr,"GLFW init failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
#if __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    int winW = 1280, winH = 720;
    GLFWwindow* win = glfwCreateWindow(winW, winH, "FLIP Fluid (C++/OpenGL)", nullptr, nullptr);
    if(!win){ fprintf(stderr,"Window create failed\n"); glfwTerminate(); return 1; }


    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        fprintf(stderr, "Failed to initialize GLAD\n");
        return -1;
    }

    // Create a global VAO for OpenGL core profile
    if (glb.globalVAO == 0) glGenVertexArrays(1, &glb.globalVAO);
    glBindVertexArray(glb.globalVAO);

    setupScene((float)winW, (float)winH);

    // ImGui setup
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // Callbacks
    glfwSetWindowUserPointer(win, nullptr);


    // Não usar callbacks de mouse do GLFW para lógica do simulador

    glfwSetKeyCallback(win, [](GLFWwindow* w, int key, int /*sc*/, int action, int /*mods*/){
        if(action==GLFW_PRESS){
            if(key==GLFW_KEY_P) scene.paused = !scene.paused;
            if(key==GLFW_KEY_M){ bool prev = scene.paused; scene.paused=false; /* simulate once below */ scene.paused = prev; }
        }
    });

    auto last = std::chrono::high_resolution_clock::now();
    while(!glfwWindowShouldClose(win)){
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> dt = now - last; last = now;
        // Fixed dt like JS
        (void)dt; // we keep scene.dt


        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // UI window
        ImGui::Begin("Simulation Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove);
        ImGui::SetWindowPos(ImVec2(10, 10), ImGuiCond_Always);
        ImGui::Checkbox("Paused", &scene.paused);
        ImGui::SliderFloat("Flip Ratio", &scene.flipRatio, 0.0f, 1.0f);
        ImGui::Checkbox("Show Particles", &scene.showParticles);
        ImGui::Checkbox("Show Grid", &scene.showGrid);
        ImGui::SliderFloat("Gravity", &scene.gravity, -20.0f, 20.0f);
        ImGui::Checkbox("Compensate Drift", &scene.compensateDrift);
        ImGui::Checkbox("Separate Particles", &scene.separateParticles);
        ImGui::End();

        // Input do mouse para o simulador (apenas se o ImGui não estiver capturando)
        ImGuiIO& io = ImGui::GetIO();
        if (!io.WantCaptureMouse) {
            // Detecta clique do mouse esquerdo
            if (glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS && !mouseDown) {
                double x, y;
                glfwGetCursorPos(win, &x, &y);
                startDrag(x, y, win);
            }
            // Detecta arrasto
            if (glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS && mouseDown) {
                double x, y;
                glfwGetCursorPos(win, &x, &y);
                drag(x, y, win);
            }
            // Detecta soltar botão
            if (glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE && mouseDown) {
                endDrag();
            }
        } else {
            // Se o ImGui capturou o mouse, sempre encerra drag do simulador
            if (mouseDown) endDrag();
        }

        if(!scene.paused){
            scene.fluid->simulate(
                scene.dt, scene.gravity, scene.flipRatio, scene.numPressureIters,
                scene.numParticleIters, scene.overRelaxation, scene.compensateDrift,
                scene.separateParticles, scene.obstacleX, scene.obstacleY, scene.obstacleRadius,
                scene.obstacleVelX, scene.obstacleVelY);
            scene.frameNr++;
        }

        draw();

        // Render ImGui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    // Cleanup ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
