cmake_minimum_required(VERSION 3.16)
project(FlipFluid)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

# Add executable
add_executable(flip_fluid flip_fluid.cpp)

# Include directories
target_include_directories(flip_fluid PRIVATE 
    glad/include
    imgui
    imgui/backends
)

# Source files for ImGui
set(IMGUI_SOURCES
    imgui/imgui.cpp
    imgui/imgui_demo.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp
    imgui/backends/imgui_impl_glfw.cpp
    imgui/backends/imgui_impl_opengl3.cpp
)

# Add ImGui sources to target
target_sources(flip_fluid PRIVATE ${IMGUI_SOURCES})

# Add GLAD source
target_sources(flip_fluid PRIVATE glad/src/glad.c)

# Link libraries
target_link_libraries(flip_fluid 
    glfw 
    OpenGL::GL
    ${CMAKE_DL_LIBS}
)

# Platform-specific settings
if(UNIX AND NOT APPLE)
    target_link_libraries(flip_fluid pthread)
endif()

if(APPLE)
    target_link_libraries(flip_fluid "-framework Cocoa" "-framework IOKit" "-framework CoreVideo")
endif()

# Compiler-specific options
if(MSVC)
    target_compile_options(flip_fluid PRIVATE /W4)
else()
    target_compile_options(flip_fluid PRIVATE -Wall -Wextra -O2)
endif()
