# A C++ port of Ten Minute Physics FLIP demo by Matthias Müller with added features

Original author: Matthias Müller  
Original demo / video: https://matthias-research.github.io/pages/tenMinutePhysics/

A C++ port of the JavaScript FLIP (Fluid-Implicit-Particle) fluid simulation using OpenGL for rendering.

## What is FLIP?

FLIP (Fluid-Implicit-Particle) is a hybrid fluid simulation method that combines the best aspects of grid-based and particle-based approaches:

- **Grid-based velocity**: Uses a regular grid (MAC grid) to solve incompressible fluid equations efficiently
- **Particle-based advection**: Particles carry velocity and other properties, avoiding numerical diffusion
- **Hybrid transfer**: Velocities are transferred between particles and grid each timestep using interpolation
- **PIC/FLIP blending**: Combines PIC (Particle-in-Cell) for stability and FLIP for detail preservation

This approach provides:
- **Incompressibility**: Proper pressure solve ensures volume conservation
- **Detail preservation**: Particles maintain fine-scale motion and vorticity
- **Stability**: Grid-based solve prevents particle clustering issues
- **Efficiency**: Faster than pure particle methods for large simulations

The simulation pipeline: integrate particles → separate particles → handle collisions → transfer to grid → solve pressure → transfer back to particles.

## Features

- FLIP/PIC fluid simulation with hybrid particle-grid method
- Interactive obstacle manipulation with mouse drag
- Real-time particle and grid visualization with color mapping
- Adjustable simulation parameters via ImGui interface
- 2D gravity vector control (X and Y components)
- Particle separation and collision handling
- Density-based drift compensation
- Cross-platform support (Linux, macOS, Windows)

## Dependencies

- **GLFW 3.x** - Window management and input
- **OpenGL 3.3+** - Graphics rendering
- **GLAD** - OpenGL loader (included)
- **Dear ImGui** - GUI interface (included)
- **C++20 compatible compiler**

## Building

### Option 1: Using CMake (Recommended)

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get install libglfw3-dev

# Install dependencies (macOS with Homebrew)
brew install glfw

# Build
mkdir build
cd build
cmake ..
make
```

### Option 2: Using Makefile

```bash
# Install dependencies first (as above)
make
```

### Option 3: Manual compilation

```bash
# Linux
g++ -std=c++20 flip_fluid.cpp glad/src/glad.c imgui/*.cpp imgui/backends/imgui_impl_*.cpp \
    -Iglad/include -Iimgui -Iimgui/backends \
    -lglfw -lGL -ldl -lpthread -o flip_fluid

# macOS
g++ -std=c++20 flip_fluid.cpp glad/src/glad.c imgui/*.cpp imgui/backends/imgui_impl_*.cpp \
    -Iglad/include -Iimgui -Iimgui/backends \
    -lglfw -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo \
    -o flip_fluid
```

### Windows (MSYS2 / MinGW and Visual Studio)

Windows has two common workflows — a MinGW/MSYS2 toolchain or the native Visual Studio/MSVC toolchain.

1) MSYS2 / MinGW (quick, command-line):

```bash
# from an MSYS2 MinGW64 shell
# install required packages once:
pacman -Syu
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-glfw mingw-w64-x86_64-toolchain

# build (adjust include paths if your imgui/glad folders are elsewhere):
g++ -std=c++17 -O2 flip_fluid.cpp glad/src/glad.c imgui/*.cpp imgui/backends/imgui_impl_*.cpp \
    -Iglad/include -Iimgui -Iimgui/backends \
    -L/mingw64/lib -lglfw3 -lopengl32 -lgdi32 -o flip_fluid.exe

# If linking fails, use pkg-config to find flags (if available):
# g++ ... $(pkg-config --cflags --libs glfw3)
```

2) Visual Studio / CMake (recommended for IDE and debugging):

```
# Generate a Visual Studio solution using CMake (from a Developer Command Prompt or PowerShell):
md build && cd build
cmake -G "Visual Studio 17 2022" ..
# or choose the generator matching your VS version
cmake --build . --config Release

# Notes:
# - Ensure GLFW (and other deps) are installed or available as prebuilt libs and adjust CMakeLists to point to them.
# - You can also add the glad and ImGui sources into the project (they're included in this repo layout) so an out-of-the-box CMake build works.

## Required Directory Structure

Make sure you have the following directories and files:
```
Fluido/
├── flip_fluid.cpp          # Main simulation code
├── CMakeLists.txt          # CMake build file
├── Makefile               # Alternative build file
├── glad/
│   ├── include/glad/
│   │   └── glad.h
│   ├── include/KHR/
│   │   └── khrplatform.h
│   └── src/
│       └── glad.c
└── imgui/
    ├── imgui.h
    ├── imgui.cpp
    ├── imgui_demo.cpp
    ├── imgui_draw.cpp
    ├── imgui_tables.cpp
    ├── imgui_widgets.cpp
    └── backends/
        ├── imgui_impl_glfw.h
        ├── imgui_impl_glfw.cpp
        ├── imgui_impl_opengl3.h
        └── imgui_impl_opengl3.cpp
```

## Getting Dependencies

### GLAD
1. Go to https://glad.dav1d.de/
2. Generate OpenGL 3.3 Core profile loader
3. Extract to `glad/` directory

### Dear ImGui
1. Download from https://github.com/ocornut/imgui
2. Extract to `imgui/` directory
3. Make sure to include the backends folder

## Controls

- **Mouse**: Drag to move the red obstacle
- **P**: Toggle pause/resume simulation
- **M**: Single step when paused
- **GUI**: Adjust parameters in real-time

## Parameters

- **Flip Ratio**: Balance between FLIP (1.0) and PIC (0.0) methods
  - FLIP: Preserves detail but can be noisy
  - PIC: More stable but introduces damping
- **Gravity X/Y**: 2D gravity vector components (default: 0, -9.81)
- **Show Particles**: Toggle particle rendering with velocity-based colors
- **Show Grid**: Toggle grid cell visualization with averaged particle colors
- **Compensate Drift**: Density correction to prevent volume loss
- **Separate Particles**: Particle collision resolution to prevent clustering
- **Pressure Iterations**: SOR solver iterations (more = better convergence)
- **Particle Iterations**: Separation algorithm iterations per frame

## Performance Notes

- Typical performance: 1000-3000 particles at 60 FPS on modern hardware
- Memory usage: ~100MB for 2000 particles with 100x100 grid
- Bottlenecks: Particle separation O(N²) in dense regions, pressure solve O(grid cells × iterations)
- Optimization opportunities: Spatial hashing, GPU compute shaders, multi-threading

## License

- MIT License (same as original JavaScript version)