#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <array>
#include <chrono>
#include <algorithm>
#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

// Dear ImGui
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

// ----------------------------- Constants -----------------------------

constexpr int U_FIELD = 0;
constexpr int V_FIELD = 1;

constexpr int FLUID_CELL = 0;
constexpr int AIR_CELL = 1;
constexpr int SOLID_CELL = 2;

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
    vec2 screenTransform = vec2(2.0/domainSize.x, 2.0/domainSize.y);
    vec2 offset = vec2(-1.0, -1.0);
    gl_Position = vec4(attrPosition * screenTransform + offset, 0.0, 1.0);
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
        vec2 coord = gl_PointCoord - vec2(0.5);
        if (dot(coord, coord) > 0.25) discard;
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
    vec2 screenTransform = vec2(2.0/domainSize.x, 2.0/domainSize.y);
    vec2 offset = vec2(-1.0, -1.0);
    gl_Position = vec4(v * screenTransform + offset, 0.0, 1.0);
    fragColor = color;
}
)GLSL";

static const char* MESH_FS = R"GLSL(
#version 330 core
in vec3 fragColor;
out vec4 outColor;
void main(){ outColor = vec4(fragColor, 1.0); }
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

static float clampf(float x, float a, float b){ 
    return std::max(a, std::min(b, x)); 
}

// -------------------------- FlipFluid --------------------------

class FlipFluid {
public:
    // Grid properties
    float density;
    int fNumX, fNumY, fNumCells;
    float h;
    float fInvSpacing;
    
    // Velocity fields
    std::vector<float> u, v, du, dv, prevU, prevV;
    std::vector<float> p, s;
    std::vector<int> cellType;
    std::vector<float> cellColor;
    // Reusable accumulators to avoid per-frame allocations in updateCellColors()
    std::vector<float> cellAccum;
    std::vector<int> cellCounts;
    
    // Particles
    int maxParticles;
    std::vector<float> particlePos;
    std::vector<float> particleColor;
    std::vector<float> particleVel;
    std::vector<float> particleDensity;
    float particleRestDensity;

    float particleRadius;
    float pInvSpacing;
    int pNumX, pNumY, pNumCells;

    std::vector<int> numCellParticles;
    std::vector<int> firstCellParticle;
    std::vector<int> cellParticleIds;

    int numParticles;

    FlipFluid(float density_, float width, float height, float spacing, float particleRadius_, int maxParticles_) {
        density = density_;
        fNumX = static_cast<int>(std::floor(width / spacing)) + 1;
        fNumY = static_cast<int>(std::floor(height / spacing)) + 1;
        h = std::max(width / fNumX, height / fNumY);
        fInvSpacing = 1.0f / h;
        fNumCells = fNumX * fNumY;
        
        // Initialize velocity fields
        u.resize(fNumCells, 0.0f);
        v.resize(fNumCells, 0.0f);
        du.resize(fNumCells, 0.0f);
        dv.resize(fNumCells, 0.0f);
        prevU.resize(fNumCells, 0.0f);
        prevV.resize(fNumCells, 0.0f);
        p.resize(fNumCells, 0.0f);
        s.resize(fNumCells, 0.0f);
        cellType.resize(fNumCells, 0);
        cellColor.resize(3 * fNumCells, 0.0f);
        // allocate reusable accumulation buffers
        cellAccum.resize(3 * fNumCells, 0.0f);
        cellCounts.resize(fNumCells, 0);
        
        // Initialize particles
        maxParticles = maxParticles_;
        particlePos.resize(2 * maxParticles, 0.0f);
        particleColor.resize(3 * maxParticles, 0.0f);
        particleVel.resize(2 * maxParticles, 0.0f);
        particleDensity.resize(fNumCells, 0.0f);
        particleRestDensity = 0.0f;
        
        // Initialize particle colors to blue
        for (int i = 0; i < maxParticles; i++) {
            particleColor[3 * i + 2] = 1.0f;
        }
        
        particleRadius = particleRadius_;
        pInvSpacing = 1.0f / (2.2f * particleRadius);
        pNumX = static_cast<int>(std::floor(width * pInvSpacing)) + 1;
        pNumY = static_cast<int>(std::floor(height * pInvSpacing)) + 1;
        pNumCells = pNumX * pNumY;
        
        numCellParticles.resize(pNumCells, 0);
        firstCellParticle.resize(pNumCells + 1, 0);
        cellParticleIds.resize(maxParticles, 0);
        
        numParticles = 0;
    }

    void integrateParticles(float dt, float gx, float gy) {
        for (int i = 0; i < numParticles; i++) {
            particleVel[2 * i] += dt * gx;
            particleVel[2 * i + 1] += dt * gy;
            particlePos[2 * i] += particleVel[2 * i] * dt;
            particlePos[2 * i + 1] += particleVel[2 * i + 1] * dt;
        }
    }

    void pushParticlesApart(int numIters) {
        float colorDiffusionCoeff = 0.001f;
        
        // count particles per cell
        std::fill(numCellParticles.begin(), numCellParticles.end(), 0);
        
        for (int i = 0; i < numParticles; i++) {
            float x = particlePos[2 * i];
            float y = particlePos[2 * i + 1];
            
            int xi = clampf(std::floor(x * pInvSpacing), 0, pNumX - 1);
            int yi = clampf(std::floor(y * pInvSpacing), 0, pNumY - 1);
            int cellNr = xi * pNumY + yi;
            numCellParticles[cellNr]++;
        }
        
        // partial sums
        int first = 0;
        for (int i = 0; i < pNumCells; i++) {
            first += numCellParticles[i];
            firstCellParticle[i] = first;
        }
        firstCellParticle[pNumCells] = first;
        
        // fill particles into cells
        for (int i = 0; i < numParticles; i++) {
            float x = particlePos[2 * i];
            float y = particlePos[2 * i + 1];
            
            int xi = clampf(std::floor(x * pInvSpacing), 0, pNumX - 1);
            int yi = clampf(std::floor(y * pInvSpacing), 0, pNumY - 1);
            int cellNr = xi * pNumY + yi;
            firstCellParticle[cellNr]--;
            cellParticleIds[firstCellParticle[cellNr]] = i;
        }
        
        // push particles apart
        float minDist = 2.0f * particleRadius;
        float minDist2 = minDist * minDist;
        
        for (int iter = 0; iter < numIters; iter++) {
            for (int i = 0; i < numParticles; i++) {
                float px = particlePos[2 * i];
                float py = particlePos[2 * i + 1];
                
                int pxi = static_cast<int>(std::floor(px * pInvSpacing));
                int pyi = static_cast<int>(std::floor(py * pInvSpacing));
                int x0 = std::max(pxi - 1, 0);
                int y0 = std::max(pyi - 1, 0);
                int x1 = std::min(pxi + 1, pNumX - 1);
                int y1 = std::min(pyi + 1, pNumY - 1);
                
                for (int xi = x0; xi <= x1; xi++) {
                    for (int yi = y0; yi <= y1; yi++) {
                        int cellNr = xi * pNumY + yi;
                        int first = firstCellParticle[cellNr];
                        int last = firstCellParticle[cellNr + 1];
                        for (int j = first; j < last; j++) {
                            int id = cellParticleIds[j];
                            if (id == i) continue;
                            
                            float qx = particlePos[2 * id];
                            float qy = particlePos[2 * id + 1];
                            
                            float dx = qx - px;
                            float dy = qy - py;
                            float d2 = dx * dx + dy * dy;
                            if (d2 > minDist2 || d2 == 0.0f) continue;
                            
                            float d = std::sqrt(d2);
                            float s = 0.5f * (minDist - d) / d;
                            dx *= s;
                            dy *= s;
                            particlePos[2 * i] -= dx;
                            particlePos[2 * i + 1] -= dy;
                            particlePos[2 * id] += dx;
                            particlePos[2 * id + 1] += dy;
                            
                            // diffuse colors
                            for (int k = 0; k < 3; k++) {
                                float color0 = particleColor[3 * i + k];
                                float color1 = particleColor[3 * id + k];
                                float color = (color0 + color1) * 0.5f;
                                particleColor[3 * i + k] = color0 + (color - color0) * colorDiffusionCoeff;
                                particleColor[3 * id + k] = color1 + (color - color1) * colorDiffusionCoeff;
                            }
                        }
                    }
                }
            }
        }
    }

    void handleParticleCollisions(float obstacleX, float obstacleY, float obstacleRadius, float obstacleVelX, float obstacleVelY) {
        float h_ = 1.0f / fInvSpacing;
        float r = particleRadius;
        float minX = h_ + r;
        float maxX = (fNumX - 1) * h_ - r;
        float minY = h_ + r;
        float maxY = (fNumY - 1) * h_ - r;

        bool obstacleActive = (obstacleRadius > 0.0f);
        float minDist = obstacleRadius + r;
        float minDist2 = minDist * minDist;

        for (int i = 0; i < numParticles; i++) {
            float x = particlePos[2 * i];
            float y = particlePos[2 * i + 1];

            if (obstacleActive) {
                float dx = x - obstacleX;
                float dy = y - obstacleY;
                float d2 = dx * dx + dy * dy;
                // obstacle collision: only when obstacle is active
                if (d2 < minDist2) {
                    particleVel[2 * i] = obstacleVelX;
                    particleVel[2 * i + 1] = obstacleVelY;
                }
            }

            // wall collisions (always applied)
            if (x < minX) {
                x = minX;
                particleVel[2 * i] = 0.0f;
            }
            if (x > maxX) {
                x = maxX;
                particleVel[2 * i] = 0.0f;
            }
            if (y < minY) {
                y = minY;
                particleVel[2 * i + 1] = 0.0f;
            }
            if (y > maxY) {
                y = maxY;
                particleVel[2 * i + 1] = 0.0f;
            }

            particlePos[2 * i] = x;
            particlePos[2 * i + 1] = y;
        }
    }

    void updateParticleDensity() {
        int n = fNumY;
        float h_ = h;
        float h1 = fInvSpacing;
        float h2 = 0.5f * h_;
        
        std::fill(particleDensity.begin(), particleDensity.end(), 0.0f);
        
        for (int i = 0; i < numParticles; i++) {
            float x = particlePos[2 * i];
            float y = particlePos[2 * i + 1];
            
            x = clampf(x, h_, (fNumX - 1) * h_);
            y = clampf(y, h_, (fNumY - 1) * h_);
            
            int x0 = static_cast<int>(std::floor((x - h2) * h1));
            float tx = ((x - h2) - x0 * h_) * h1;
            int x1 = std::min(x0 + 1, fNumX - 2);
            
            int y0 = static_cast<int>(std::floor((y - h2) * h1));
            float ty = ((y - h2) - y0 * h_) * h1;
            int y1 = std::min(y0 + 1, fNumY - 2);
            
            float sx = 1.0f - tx;
            float sy = 1.0f - ty;
            
            if (x0 < fNumX && y0 < fNumY) particleDensity[x0 * n + y0] += sx * sy;
            if (x1 < fNumX && y0 < fNumY) particleDensity[x1 * n + y0] += tx * sy;
            if (x1 < fNumX && y1 < fNumY) particleDensity[x1 * n + y1] += tx * ty;
            if (x0 < fNumX && y1 < fNumY) particleDensity[x0 * n + y1] += sx * ty;
        }
        
        if (particleRestDensity == 0.0f) {
            float sum = 0.0f;
            int numFluidCells = 0;
            
            for (int i = 0; i < fNumCells; i++) {
                if (cellType[i] == FLUID_CELL) {
                    sum += particleDensity[i];
                    numFluidCells++;
                }
            }
            
            if (numFluidCells > 0) {
                particleRestDensity = sum / numFluidCells;
            }
        }
    }

    void transferVelocities(bool toGrid, float flipRatio = 0.0f) {
        int n = fNumY;
        float h_ = h;
        float h1 = fInvSpacing;
        float h2 = 0.5f * h_;
        
        if (toGrid) {
            prevU = u;
            prevV = v;
            
            std::fill(du.begin(), du.end(), 0.0f);
            std::fill(dv.begin(), dv.end(), 0.0f);
            std::fill(u.begin(), u.end(), 0.0f);
            std::fill(v.begin(), v.end(), 0.0f);
            
            for (int i = 0; i < fNumCells; i++) {
                cellType[i] = (s[i] == 0.0f) ? SOLID_CELL : AIR_CELL;
            }
            
            for (int i = 0; i < numParticles; i++) {
                float x = particlePos[2 * i];
                float y = particlePos[2 * i + 1];
                int xi = clampf(std::floor(x * h1), 0, fNumX - 1);
                int yi = clampf(std::floor(y * h1), 0, fNumY - 1);
                int cellNr = xi * n + yi;
                if (cellType[cellNr] == AIR_CELL) {
                    cellType[cellNr] = FLUID_CELL;
                }
            }
        }
        
        for (int component = 0; component < 2; component++) {
            float dx = (component == 0) ? 0.0f : h2;
            float dy = (component == 0) ? h2 : 0.0f;
            
            std::vector<float>& f = (component == 0) ? u : v;
            std::vector<float>& prevF = (component == 0) ? prevU : prevV;
            std::vector<float>& d = (component == 0) ? du : dv;
            
            for (int i = 0; i < numParticles; i++) {
                float x = particlePos[2 * i];
                float y = particlePos[2 * i + 1];
                
                x = clampf(x, h_, (fNumX - 1) * h_);
                y = clampf(y, h_, (fNumY - 1) * h_);
                
                int x0 = std::min(static_cast<int>(std::floor((x - dx) * h1)), fNumX - 2);
                float tx = ((x - dx) - x0 * h_) * h1;
                int x1 = std::min(x0 + 1, fNumX - 2);
                
                int y0 = std::min(static_cast<int>(std::floor((y - dy) * h1)), fNumY - 2);
                float ty = ((y - dy) - y0 * h_) * h1;
                int y1 = std::min(y0 + 1, fNumY - 2);
                
                float sx = 1.0f - tx;
                float sy = 1.0f - ty;
                
                float d0 = sx * sy;
                float d1 = tx * sy;
                float d2 = tx * ty;
                float d3 = sx * ty;
                
                int nr0 = x0 * n + y0;
                int nr1 = x1 * n + y0;
                int nr2 = x1 * n + y1;
                int nr3 = x0 * n + y1;
                
                if (toGrid) {
                    float pv = particleVel[2 * i + component];
                    f[nr0] += pv * d0; d[nr0] += d0;
                    f[nr1] += pv * d1; d[nr1] += d1;
                    f[nr2] += pv * d2; d[nr2] += d2;
                    f[nr3] += pv * d3; d[nr3] += d3;
                } else {
                    int offset = (component == 0) ? n : 1;
                    float valid0 = (cellType[nr0] != AIR_CELL || cellType[nr0 - offset] != AIR_CELL) ? 1.0f : 0.0f;
                    float valid1 = (cellType[nr1] != AIR_CELL || cellType[nr1 - offset] != AIR_CELL) ? 1.0f : 0.0f;
                    float valid2 = (cellType[nr2] != AIR_CELL || cellType[nr2 - offset] != AIR_CELL) ? 1.0f : 0.0f;
                    float valid3 = (cellType[nr3] != AIR_CELL || cellType[nr3 - offset] != AIR_CELL) ? 1.0f : 0.0f;
                    
                    float v = particleVel[2 * i + component];
                    float totalWeight = valid0 * d0 + valid1 * d1 + valid2 * d2 + valid3 * d3;
                    
                    if (totalWeight > 0.0f) {
                        float picV = (valid0 * d0 * f[nr0] + valid1 * d1 * f[nr1] + 
                                     valid2 * d2 * f[nr2] + valid3 * d3 * f[nr3]) / totalWeight;
                        float corr = (valid0 * d0 * (f[nr0] - prevF[nr0]) + valid1 * d1 * (f[nr1] - prevF[nr1]) +
                                     valid2 * d2 * (f[nr2] - prevF[nr2]) + valid3 * d3 * (f[nr3] - prevF[nr3])) / totalWeight;
                        float flipV = v + corr;
                        
                        particleVel[2 * i + component] = (1.0f - flipRatio) * picV + flipRatio * flipV;
                    }
                }
            }
            
            if (toGrid) {
                for (int i = 0; i < static_cast<int>(f.size()); i++) {
                    if (d[i] > 0.0f) {
                        f[i] /= d[i];
                    }
                }
                
                // restore solid cells
                for (int i = 0; i < fNumX; i++) {
                    for (int j = 0; j < fNumY; j++) {
                        bool solid = (cellType[i * n + j] == SOLID_CELL);
                        if (solid || (i > 0 && cellType[(i - 1) * n + j] == SOLID_CELL)) {
                            u[i * n + j] = prevU[i * n + j];
                        }
                        if (solid || (j > 0 && cellType[i * n + j - 1] == SOLID_CELL)) {
                            v[i * n + j] = prevV[i * n + j];
                        }
                    }
                }
            }
        }
    }

    void solveIncompressibility(int numIters, float dt, float overRelaxation, bool compensateDrift = true) {
        std::fill(p.begin(), p.end(), 0.0f);
        prevU = u;
        prevV = v;
        
        int n = fNumY;
        float cp = density * h / dt;
        
        for (int iter = 0; iter < numIters; iter++) {
            for (int i = 1; i < fNumX - 1; i++) {
                for (int j = 1; j < fNumY - 1; j++) {
                    if (cellType[i * n + j] != FLUID_CELL) continue;
                    
                    int center = i * n + j;
                    int left = (i - 1) * n + j;
                    int right = (i + 1) * n + j;
                    int bottom = i * n + j - 1;
                    int top = i * n + j + 1;
                    
                    float sx0 = s[left];
                    float sx1 = s[right];
                    float sy0 = s[bottom];
                    float sy1 = s[top];
                    float sTotal = sx0 + sx1 + sy0 + sy1;
                    
                    if (sTotal == 0.0f) continue;
                    
                    float div = u[right] - u[center] + v[top] - v[center];
                    
                    if (particleRestDensity > 0.0f && compensateDrift) {
                        float k = 1.0f;
                        float compression = particleDensity[i * n + j] - particleRestDensity;
                        if (compression > 0.0f) {
                            div = div - k * compression;
                        }
                    }
                    
                    float pressure = -div / sTotal;
                    pressure *= overRelaxation;
                    p[center] += cp * pressure;
                    
                    u[center] -= sx0 * pressure;
                    u[right] += sx1 * pressure;
                    v[center] -= sy0 * pressure;
                    v[top] += sy1 * pressure;
                }
            }
        }
    }

    void updateParticleColors() {
        float h1 = fInvSpacing;
        
        for (int i = 0; i < numParticles; i++) {
            float s = 0.01f;
            
            particleColor[3 * i] = clampf(particleColor[3 * i] - s, 0.0f, 1.0f);
            particleColor[3 * i + 1] = clampf(particleColor[3 * i + 1] - s, 0.0f, 1.0f);
            particleColor[3 * i + 2] = clampf(particleColor[3 * i + 2] + s, 0.0f, 1.0f);
            
            float x = particlePos[2 * i];
            float y = particlePos[2 * i + 1];
            int xi = clampf(std::floor(x * h1), 1, fNumX - 1);
            int yi = clampf(std::floor(y * h1), 1, fNumY - 1);
            int cellNr = xi * fNumY + yi;
            
            float d0 = particleRestDensity;
            
            if (d0 > 0.0f) {
                float relDensity = particleDensity[cellNr] / d0;
                if (relDensity < 0.7f) {
                    float s = 0.8f;
                    particleColor[3 * i] = s;
                    particleColor[3 * i + 1] = s;
                    particleColor[3 * i + 2] = 1.0f;
                }
            }
        }
    }

    void setSciColor(int cellNr, float val, float minVal, float maxVal) {
        val = std::min(std::max(val, minVal), maxVal - 0.0001f);
        float d = maxVal - minVal;
        val = (d == 0.0f) ? 0.5f : (val - minVal) / d;
        float m = 0.25f;
        int num = static_cast<int>(std::floor(val / m));
        float s = (val - num * m) / m;
        float r, g, b;
        
        switch (num) {
            case 0: r = 0.0f; g = s; b = 1.0f; break;
            case 1: r = 0.0f; g = 1.0f; b = 1.0f - s; break;
            case 2: r = s; g = 1.0f; b = 0.0f; break;
            case 3: r = 1.0f; g = 1.0f - s; b = 0.0f; break;
            default: r = 1.0f; g = 0.0f; b = 0.0f; break;
        }
        
        cellColor[3 * cellNr] = r;
        cellColor[3 * cellNr + 1] = g;
        cellColor[3 * cellNr + 2] = b;
    }

    void updateCellColors() {
        // Compute average particle colors per grid cell using preallocated buffers to avoid allocations
        float h1 = fInvSpacing;

        // reset accumulators
        std::fill(cellAccum.begin(), cellAccum.end(), 0.0f);
        std::fill(cellCounts.begin(), cellCounts.end(), 0);
        
        for (int i = 0; i < numParticles; i++) {
            float x = particlePos[2 * i];
            float y = particlePos[2 * i + 1];
            int xi = static_cast<int>(x * h1);
            int yi = static_cast<int>(y * h1);
            xi = std::max(0, std::min(xi, fNumX - 1));
            yi = std::max(0, std::min(yi, fNumY - 1));
            int cellNr = xi * fNumY + yi;
            cellCounts[cellNr]++;
            cellAccum[3 * cellNr + 0] += particleColor[3 * i + 0];
            cellAccum[3 * cellNr + 1] += particleColor[3 * i + 1];
            cellAccum[3 * cellNr + 2] += particleColor[3 * i + 2];
        }

        // Clear cell colors
        std::fill(cellColor.begin(), cellColor.end(), 0.0f);

        for (int i = 0; i < fNumCells; i++) {
            if (cellType[i] == SOLID_CELL) {
                cellColor[3 * i] = 0.5f;
                cellColor[3 * i + 1] = 0.5f;
                cellColor[3 * i + 2] = 0.5f;
            } else if (cellType[i] == FLUID_CELL) {
                if (cellCounts[i] > 0) {
                    float inv = 1.0f / static_cast<float>(cellCounts[i]);
                    float r = cellAccum[3 * i + 0] * inv;
                    float g = cellAccum[3 * i + 1] * inv;
                    float b = cellAccum[3 * i + 2] * inv;
                    // Clamp just in case
                    cellColor[3 * i] = std::min(std::max(r, 0.0f), 1.0f);
                    cellColor[3 * i + 1] = std::min(std::max(g, 0.0f), 1.0f);
                    cellColor[3 * i + 2] = std::min(std::max(b, 0.0f), 1.0f);
                } else {
                    // No particles in this fluid cell: fall back to density-based scientific color
                    float d = particleDensity[i];
                    if (particleRestDensity > 0.0f) {
                        d /= particleRestDensity;
                    }
                    setSciColor(i, d, 0.0f, 2.0f);
                }
            } else {
                // AIR cell with no particles: leave transparent/black (already zero)
            }
        }
    }

    void simulate(float dt, float gx, float gy, float flipRatio, int numPressureIters, int numParticleIters,
                  float overRelaxation, bool compensateDrift, bool separateParticles,
                  float obstacleX, float obstacleY, float obstacleRadius, float obstacleVelX, float obstacleVelY) {
        int numSubSteps = 1;
        float sdt = dt / numSubSteps;
        
        for (int step = 0; step < numSubSteps; step++) {
            integrateParticles(sdt, gx, gy);
            if (separateParticles) {
                pushParticlesApart(numParticleIters);
            }
            handleParticleCollisions(obstacleX, obstacleY, obstacleRadius, obstacleVelX, obstacleVelY);
            transferVelocities(true);
            updateParticleDensity();
            solveIncompressibility(numPressureIters, sdt, overRelaxation, compensateDrift);
            transferVelocities(false, flipRatio);
        }
        
        updateParticleColors();
        updateCellColors();
    }
};

// --------------------------- Scene state ---------------------------

struct Scene {
    float gravityX = 0.0f;
    float gravityY = -9.81f;
    float dt = 1.0f / 60.0f;
    float flipRatio = 0.9f;
    int numPressureIters = 25;
    int numParticleIters = 2;
    int frameNr = 0;
    float overRelaxation = 1.9f;
    bool compensateDrift = true;
    bool separateParticles = true;
    float obstacleX = 0.0f, obstacleY = 0.0f, obstacleRadius = 0.15f;
    bool paused = false;
    bool showObstacle = false; // collisions di400sabled until user clicks
    // Separate flag to control only rendering of the obstacle. Keep collisions using showObstacle.
    bool renderObstacle = false;
    float obstacleVelX = 0.0f, obstacleVelY = 0.0f;
    bool showParticles = false;
    bool showGrid = true;
    FlipFluid* fluid = nullptr;
    
    float simWidth = 0.0f, simHeight = 0.0f;
} scene;

// ---------------------------- OpenGL Objects ----------------------------

struct GLObjects {
    GLuint pointProg = 0, meshProg = 0;
    GLuint gridVBO = 0, gridColorVBO = 0;
    GLuint particleVBO = 0, particleColorVBO = 0;
    GLuint diskVBO = 0, diskEBO = 0;
    GLuint globalVAO = 0;
    // VAOs to store vertex attribute state so we don't reconfigure every frame
    GLuint gridVAO = 0, particleVAO = 0, diskVAO = 0;

    // Cached uniform locations for point shader
    GLint point_dom_loc = -1;
    GLint point_psize_loc = -1;
    GLint point_drawDisk_loc = -1;

    // Cached uniform locations for mesh shader
    GLint mesh_dom_loc = -1;
    GLint mesh_color_loc = -1;
    GLint mesh_tr_loc = -1;
    GLint mesh_sc_loc = -1;
} glb;

// --- Buffer creation helpers ---
void createOrUpdateGridVBO() {
    auto& f = *scene.fluid;
    if (glb.gridVBO == 0) glGenBuffers(1, &glb.gridVBO);
    if (glb.gridColorVBO == 0) {
        glGenBuffers(1, &glb.gridColorVBO);
        // allocate buffer once for dynamic updates
        glBindBuffer(GL_ARRAY_BUFFER, glb.gridColorVBO);
        glBufferData(GL_ARRAY_BUFFER, f.cellColor.size() * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    
    static std::vector<float> cellCenters; // persistent to avoid reallocations
    cellCenters.resize(2 * f.fNumCells);
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

    // Setup/grid VAO once
    if (glb.gridVAO == 0) {
        glGenVertexArrays(1, &glb.gridVAO);
        glBindVertexArray(glb.gridVAO);

        glBindBuffer(GL_ARRAY_BUFFER, glb.gridVBO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);

        glBindBuffer(GL_ARRAY_BUFFER, glb.gridColorVBO);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
}

void createParticleVBOs() {
    if (glb.particleVBO == 0) glGenBuffers(1, &glb.particleVBO);
    if (glb.particleColorVBO == 0) glGenBuffers(1, &glb.particleColorVBO);
    // allocate once to the maximum size to avoid repeated allocations; actual updates will use glBufferSubData
    if (scene.fluid) {
        auto& f = *scene.fluid;
        glBindBuffer(GL_ARRAY_BUFFER, glb.particleVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2 * f.maxParticles, nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, glb.particleColorVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * f.maxParticles, nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    // Setup particle VAO once
    if (glb.particleVAO == 0) {
        glGenVertexArrays(1, &glb.particleVAO);
        glBindVertexArray(glb.particleVAO);

        glBindBuffer(GL_ARRAY_BUFFER, glb.particleVBO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);

        glBindBuffer(GL_ARRAY_BUFFER, glb.particleColorVBO);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
}

// Create disk mesh buffers (triangle fan) for obstacle rendering
void createDiskBuffers() {
    const int numSegs = 50;
    if (glb.diskVBO == 0) {
        glGenBuffers(1, &glb.diskVBO);
        std::vector<float> diskVerts(2 * (numSegs + 1));
        diskVerts[0] = 0.0f; diskVerts[1] = 0.0f;
        float dphi = 2.0f * static_cast<float>(M_PI) / numSegs;
        for (int i = 0; i < numSegs; i++) {
            diskVerts[2 * (i + 1)] = std::cos(i * dphi);
            diskVerts[2 * (i + 1) + 1] = std::sin(i * dphi);
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
            diskIds[p++] = 1 + (i + 1) % numSegs;
        }
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glb.diskEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, diskIds.size() * sizeof(unsigned short), diskIds.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }

    // Setup disk VAO once
    if (glb.diskVAO == 0) {
        glGenVertexArrays(1, &glb.diskVAO);
        glBindVertexArray(glb.diskVAO);

        glBindBuffer(GL_ARRAY_BUFFER, glb.diskVBO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);

        // bind element buffer to VAO
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glb.diskEBO);

        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }
}

// --------------------------- Setup scene ---------------------------

void setObstacle(FlipFluid& f, float x, float y, bool reset) {
    float vx = 0.0f, vy = 0.0f;
    if (!reset) {
        vx = (x - scene.obstacleX) / scene.dt;
        vy = (y - scene.obstacleY) / scene.dt;
    }
    scene.obstacleX = x;
    scene.obstacleY = y;
    float r = scene.obstacleRadius;
    int n = f.fNumY;
    
    for (int i = 1; i < f.fNumX - 2; i++) {
        for (int j = 1; j < f.fNumY - 2; j++) {
            f.s[i * n + j] = 1.0f;
            
            float dx = (i + 0.5f) * f.h - x;
            float dy = (j + 0.5f) * f.h - y;
            
            if (dx * dx + dy * dy < r * r) {
                f.s[i * n + j] = 0.0f;
                f.u[i * n + j] = vx;
                f.u[(i + 1) * n + j] = vx;
                f.v[i * n + j] = vy;
                f.v[i * n + j + 1] = vy;
            }
        }
    }
    
    scene.showObstacle = true;
    scene.obstacleVelX = vx;
    scene.obstacleVelY = vy;
}

// Remove any obstacle from the grid (restore interior cells to fluid)
void removeObstacle(FlipFluid& f) {
    int n = f.fNumY;
    // mark cells that currently are obstacle (solid) but not the tank walls
    std::vector<char> wasObstacle(f.fNumCells, 0);
    for (int i = 1; i < f.fNumX - 1; ++i) {
        for (int j = 1; j < f.fNumY - 1; ++j) {
            int idx = i * n + j;
            if (f.s[idx] == 0.0f) wasObstacle[idx] = 1;
        }
    }

    // restore s: walls remain solid, interior become fluid
    for (int i = 0; i < f.fNumX; ++i) {
        for (int j = 0; j < f.fNumY; ++j) {
            int idx = i * n + j;
            if (i == 0 || i == f.fNumX - 1 || j == 0 || j == f.fNumY - 1) {
                f.s[idx] = 0.0f; // wall
            } else {
                f.s[idx] = 1.0f; // fluid
            }
        }
    }

    // For cells that were obstacle, repair velocity fields by averaging neighbor velocities
    for (int i = 1; i < f.fNumX - 1; ++i) {
        for (int j = 1; j < f.fNumY - 1; ++j) {
            int idx = i * n + j;
            if (!wasObstacle[idx]) continue;

            float sumU = 0.0f, sumV = 0.0f;
            int cnt = 0;
            const int ni[4] = { -1, 1, 0, 0 };
            const int nj[4] = { 0, 0, -1, 1 };
            for (int k = 0; k < 4; ++k) {
                int ii = i + ni[k];
                int jj = j + nj[k];
                int nidx = ii * n + jj;
                if (f.s[nidx] != 0.0f) { // not solid
                    sumU += f.u[nidx];
                    sumV += f.v[nidx];
                    cnt++;
                }
            }
            float avgU = (cnt > 0) ? (sumU / cnt) : 0.0f;
            float avgV = (cnt > 0) ? (sumV / cnt) : 0.0f;
            f.u[idx] = avgU;
            f.v[idx] = avgV;
            f.prevU[idx] = avgU;
            f.prevV[idx] = avgV;
            f.du[idx] = 0.0f;
            f.dv[idx] = 0.0f;
        }
    }

    scene.showObstacle = false;
    scene.obstacleVelX = scene.obstacleVelY = 0.0f;
}

void setupScene(float canvasWidth, float canvasHeight, int res) {
    float simHeight = 3.0f;
    float cScale = canvasHeight / simHeight;
    float simWidth = canvasWidth / cScale;
    scene.simWidth = simWidth;
    scene.simHeight = simHeight;
    
    scene.obstacleRadius = 0.15f;
    scene.overRelaxation = 1.9f;
    scene.dt = 1.0f / 60.0f;
    scene.numPressureIters = 50;
    scene.numParticleIters = 2;
    
    float tankHeight = 1.0f * simHeight;
    float tankWidth = 1.0f * simWidth;
    float h = tankHeight / res;
    float density = 1000.0f;
    
    float relWaterHeight = 0.8f;
    float relWaterWidth = 0.6f;
    
    // compute number of particles
    float r = 0.3f * h;
    float dx = 2.0f * r;
    float dy = std::sqrt(3.0f) / 2.0f * dx;
    
    int numX = static_cast<int>(std::floor((relWaterWidth * tankWidth - 2.0f * h - 2.0f * r) / dx));
    int numY = static_cast<int>(std::floor((relWaterHeight * tankHeight - 2.0f * h - 2.0f * r) / dy));
    int maxParticles = numX * numY;
    
    // create fluid
    auto* f = new FlipFluid(density, tankWidth, tankHeight, h, r, maxParticles);
    scene.fluid = f;
    
    // create particles
    f->numParticles = numX * numY;
    int p = 0;
    for (int i = 0; i < numX; i++) {
        for (int j = 0; j < numY; j++) {
            f->particlePos[p++] = h + r + dx * i + (j % 2 == 0 ? 0.0f : r);
            f->particlePos[p++] = h + r + dy * j;
        }
    }
    
    // setup grid cells for tank
    int n = f->fNumY;
    for (int i = 0; i < f->fNumX; i++) {
        for (int j = 0; j < f->fNumY; j++) {
            float s = 1.0f; // fluid
            if (i == 0 || i == f->fNumX - 1 || j == 0 || j == f->fNumY - 1) {
                s = 0.0f; // solid (walls, floor, and roof)
            }
            f->s[i * n + j] = s;
        }
    }
    
    // no initial obstacle; obstacle will be created when the user clicks
}

// ----------------------------- Drawing -----------------------------

void draw() {
    int width = 0, height = 0;
    GLFWwindow* win = glfwGetCurrentContext();
    if (win) {
        glfwGetFramebufferSize(win, &width, &height);
        glViewport(0, 0, width, height);
    }
    
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glBindVertexArray(glb.globalVAO);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);
    
    if (glb.pointProg == 0) {
        glb.pointProg = linkProgram(POINT_VS, POINT_FS);
        // cache uniform locations
        glUseProgram(glb.pointProg);
        glb.point_dom_loc = glGetUniformLocation(glb.pointProg, "domainSize");
        glb.point_psize_loc = glGetUniformLocation(glb.pointProg, "pointSize");
        glb.point_drawDisk_loc = glGetUniformLocation(glb.pointProg, "drawDisk");
    }
    if (glb.meshProg == 0) {
        glb.meshProg = linkProgram(MESH_VS, MESH_FS);
        // cache uniform locations
        glUseProgram(glb.meshProg);
        glb.mesh_dom_loc = glGetUniformLocation(glb.meshProg, "domainSize");
        glb.mesh_color_loc = glGetUniformLocation(glb.meshProg, "color");
        glb.mesh_tr_loc = glGetUniformLocation(glb.meshProg, "translation");
        glb.mesh_sc_loc = glGetUniformLocation(glb.meshProg, "scale");
    }
    
    // Grid
    if (glb.gridVBO == 0 || glb.gridVAO == 0) createOrUpdateGridVBO();
    if (scene.showGrid) {
        auto& f = *scene.fluid;
        glUseProgram(glb.pointProg);
        glUniform2f(glb.point_dom_loc, scene.simWidth, scene.simHeight);
        // Compute a consistent pixel scale from simulation units to screen pixels
        float pixelScale = std::min(static_cast<float>(width) / scene.simWidth, static_cast<float>(height) / scene.simHeight);
        float pointSize = 0.8f * f.h * pixelScale;
        glUniform1f(glb.point_psize_loc, pointSize);
        glUniform1f(glb.point_drawDisk_loc, 0.0f);
        
        // Update colors and draw using VAO
        glBindBuffer(GL_ARRAY_BUFFER, glb.gridColorVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, scene.fluid->cellColor.size() * sizeof(float), scene.fluid->cellColor.data());
        glBindVertexArray(glb.gridVAO);
        glDrawArrays(GL_POINTS, 0, scene.fluid->fNumCells);
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    
    // Particles
    if (scene.showParticles) {
        auto& f = *scene.fluid;
        glUseProgram(glb.pointProg);
        glUniform2f(glb.point_dom_loc, scene.simWidth, scene.simHeight);
        // Use the same pixelScale approach for particle sizes so they match grid scaling
        float pixelScale = std::min(static_cast<float>(width) / scene.simWidth, static_cast<float>(height) / scene.simHeight);
        float pointSize = 2.0f * f.particleRadius * pixelScale;
        glUniform1f(glb.point_psize_loc, pointSize);
        glUniform1f(glb.point_drawDisk_loc, 1.0f);
        
        createParticleVBOs();
        // update buffers and draw using VAO
        glBindBuffer(GL_ARRAY_BUFFER, glb.particleVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, f.numParticles * 2 * sizeof(float), f.particlePos.data());
        glBindBuffer(GL_ARRAY_BUFFER, glb.particleColorVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, f.numParticles * 3 * sizeof(float), f.particleColor.data());
        glBindVertexArray(glb.particleVAO);
        glDrawArrays(GL_POINTS, 0, f.numParticles);
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    
    // Obstacle disk: render only if renderObstacle is true. Collision state still uses showObstacle
    if (scene.renderObstacle) {
        createDiskBuffers();
        glUseProgram(glb.meshProg);
        glUniform2f(glb.mesh_dom_loc, scene.simWidth, scene.simHeight);
        glUniform3f(glb.mesh_color_loc, 1.0f, 0.0f, 0.0f);
        glUniform2f(glb.mesh_tr_loc, scene.obstacleX, scene.obstacleY);
        glUniform1f(glb.mesh_sc_loc, scene.obstacleRadius + scene.fluid->particleRadius);
         
        glBindVertexArray(glb.diskVAO);
        glDrawElements(GL_TRIANGLES, 3 * 50, GL_UNSIGNED_SHORT, 0);
        glBindVertexArray(0);
    }
}

// ----------------------------- Input -----------------------------

static bool mouseDown = false;

// Helper: whether ImGui wants/should capture the mouse (including hover)
static bool imguiWantsMouse() {
    ImGuiIO& io = ImGui::GetIO();
    // Consider ImGui wanting the mouse if it explicitly requests capture
    // or if any ImGui window or item is hovered. This makes hovering the
    // UI transfer focus to ImGui so the obstacle isn't moved while hovering.
    return io.WantCaptureMouse || ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow) || ImGui::IsAnyItemHovered();
}

void startDrag(double x, double y, GLFWwindow* win) {
    int w, h;
    glfwGetFramebufferSize(win, &w, &h);
    float mx = static_cast<float>(x);
    float my = static_cast<float>(y);
    float cScale = h / scene.simHeight;
    float sx = mx / cScale;
    float sy = (h - my) / cScale;
    mouseDown = true;
    setObstacle(*scene.fluid, sx, sy, true);
    scene.paused = false;
}

void drag(double x, double y, GLFWwindow* win) {
    if (!mouseDown) return;
    int w, h;
    glfwGetFramebufferSize(win, &w, &h);
    float mx = static_cast<float>(x);
    float my = static_cast<float>(y);
    float cScale = h / scene.simHeight;
    float sx = mx / cScale;
    float sy = (h - my) / cScale;
    setObstacle(*scene.fluid, sx, sy, false);
}

void endDrag() {
    mouseDown = false;
    // remove obstacle when mouse released so collisions only occur while clicking
    if (scene.fluid) removeObstacle(*scene.fluid);
}

// ------------------------------ Main ------------------------------

int main() {
    if (!glfwInit()) {
        fprintf(stderr, "GLFW init failed\n");
        return 1;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    
    int winW = 1280, winH = 720;

    // Prompt user for desired simulation (window) size before creating the window
    std::cout << "Enter simulation window width in pixels (default 1280): ";
    std::string inW; std::getline(std::cin, inW);
    if (!inW.empty()) {
        try { winW = std::max(100, std::stoi(inW)); } catch(...) { /* keep default */ }
    }
    std::cout << "Enter simulation window height in pixels (default 720): ";
    std::string inH; std::getline(std::cin, inH);
    if (!inH.empty()) {
        try { winH = std::max(100, std::stoi(inH)); } catch(...) { /* keep default */ }
    }

    // Prompt user for grid resolution (number of cells along height)
    int gridRes = 100;
    std::cout << "Enter grid resolution (cells along height, default 100): ";
    std::string inRes; std::getline(std::cin, inRes);
    if (!inRes.empty()) {
        try { gridRes = std::max(10, std::min(2000, std::stoi(inRes))); } catch(...) { /* keep default */ }
    }

    GLFWwindow* win = glfwCreateWindow(winW, winH, "FLIP Fluid (C++/OpenGL)", nullptr, nullptr);
    if (!win) {
        fprintf(stderr, "Window create failed\n");
        glfwTerminate();
        return 1;
    }
    
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);
    
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        fprintf(stderr, "Failed to initialize GLAD\n");
        return -1;
    }
    
    if (glb.globalVAO == 0) glGenVertexArrays(1, &glb.globalVAO);
    glBindVertexArray(glb.globalVAO);
    
    setupScene(static_cast<float>(winW), static_cast<float>(winH), gridRes);
    
    // ImGui setup
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    
    // Mouse callbacks
    glfwSetMouseButtonCallback(win, [](GLFWwindow* w, int button, int action, int mods) {
        // Forward event to ImGui first so io.WantCaptureMouse is updated
        ImGui_ImplGlfw_MouseButtonCallback(w, button, action, mods);

        // Always allow ImGui to handle releases so we don't get stuck in drag state
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            if (action == GLFW_RELEASE) {
                endDrag();
                return;
            }
            // On press, only start dragging if ImGui is not hovered and doesn't want the mouse
            if (action == GLFW_PRESS) {
                ImGuiIO& io = ImGui::GetIO();
                if (io.WantCaptureMouse)
                    return;
                double x, y;
                glfwGetCursorPos(w, &x, &y);
                startDrag(x, y, w);
            }
        }
    });
    glfwSetCursorPosCallback(win, [](GLFWwindow* w, double x, double y) {
        // Forward cursor position to ImGui first
        ImGui_ImplGlfw_CursorPosCallback(w, x, y);

        ImGuiIO& io = ImGui::GetIO();
        // If ImGui currently wants the mouse (hover or capture) do not forward drag events
        if (io.WantCaptureMouse)
            return;

        if (mouseDown) {
            drag(x, y, w);
        }
    });
    
    glfwSetKeyCallback(win, [](GLFWwindow* w, int key, int /*sc*/, int action, int /*mods*/) {
        if (action == GLFW_PRESS) {
            switch (key) {
                case GLFW_KEY_P: scene.paused = !scene.paused; break;
                case GLFW_KEY_M: 
                    scene.paused = false;
                    scene.fluid->simulate(scene.dt, scene.gravityX, scene.gravityY, scene.flipRatio, 
                                        scene.numPressureIters, scene.numParticleIters,
                                        scene.overRelaxation, scene.compensateDrift, 
                                        scene.separateParticles, scene.obstacleX, 
                                        scene.obstacleY, (scene.showObstacle ? scene.obstacleRadius : 0.0f),
                                        scene.obstacleVelX, scene.obstacleVelY);
                    scene.frameNr++;
                    scene.paused = true;
                    break;
            }
        }
    });
    
    auto last = std::chrono::high_resolution_clock::now();
    while (!glfwWindowShouldClose(win)) {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> dt = now - last;
        last = now;
        
        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        // UI window
        ImGui::Begin("Simulation Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove);
        ImGui::SetWindowPos(ImVec2(10, 10), ImGuiCond_Always);
        ImGui::Checkbox("Paused", &scene.paused);
        ImGui::SliderFloat("Simulation Speed", &scene.dt, 0.001f, 0.1f, "%.4f");
        ImGui::SliderFloat("Flip Ratio", &scene.flipRatio, 0.0f, 1.0f);
        ImGui::Checkbox("Show Particles", &scene.showParticles);
        ImGui::Checkbox("Show Grid", &scene.showGrid);
        ImGui::Checkbox("Render Obstacle", &scene.renderObstacle);
        ImGui::SliderFloat("Gravity X", &scene.gravityX, -20.0f, 20.0f);
        ImGui::SliderFloat("Gravity Y", &scene.gravityY, -20.0f, 20.0f);
        ImGui::Checkbox("Compensate Drift", &scene.compensateDrift);
        ImGui::Checkbox("Separate Particles", &scene.separateParticles);
        ImGui::End();
        
        if (!scene.paused) {
            scene.fluid->simulate(scene.dt, scene.gravityX, scene.gravityY, scene.flipRatio, 
                                scene.numPressureIters, scene.numParticleIters,
                                scene.overRelaxation, scene.compensateDrift, 
                                scene.separateParticles, scene.obstacleX, 
                                scene.obstacleY, (scene.showObstacle ? scene.obstacleRadius : 0.0f),
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
    
    delete scene.fluid;
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
