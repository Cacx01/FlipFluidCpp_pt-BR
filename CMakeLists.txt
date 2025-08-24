cmake_minimum_required(VERSION 3.16)
project(FlipFluid)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

# Add executable
add_executable(flip_fluid flip_fluid.cpp)

# Locate ImGui sources in common locations (repo may contain imgui/ or imgui-1.92.2b/)
set(IMGUI_CANDIDATES
    "${CMAKE_SOURCE_DIR}/imgui"
    "${CMAKE_SOURCE_DIR}/imgui-1.92.2b"
)
set(IMGUI_DIR "")
foreach(cand IN LISTS IMGUI_CANDIDATES)
    if(EXISTS "${cand}/imgui.cpp")
        set(IMGUI_DIR ${cand})
        break()
    endif()
endforeach()

if(IMGUI_DIR STREQUAL "")
    message(FATAL_ERROR "ImGui sources not found. Place ImGui sources in ./imgui or ./imgui-1.92.2b")
endif()

# locate backend directory (common layouts)
set(IMGUI_BACKEND_DIR "${IMGUI_DIR}/backends")
if(NOT EXISTS "${IMGUI_BACKEND_DIR}/imgui_impl_glfw.cpp")
    # try examples folder (older imgui layouts)
    if(EXISTS "${IMGUI_DIR}/examples/example_glfw_opengl3")
        set(IMGUI_BACKEND_DIR "${IMGUI_DIR}/examples/example_glfw_opengl3")
    endif()
endif()

# Include directories
target_include_directories(flip_fluid PRIVATE 
    glad/include
    ${IMGUI_DIR}
    ${IMGUI_BACKEND_DIR}
)

# Source files for ImGui (explicit paths)
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_BACKEND_DIR}/imgui_impl_glfw.cpp
    ${IMGUI_BACKEND_DIR}/imgui_impl_opengl3.cpp
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
