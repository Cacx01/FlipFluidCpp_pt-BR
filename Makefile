# Simple Makefile for FLIP Fluid simulation
# Requires: GLFW, OpenGL, GLAD, ImGui

CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O2 -g
TARGET = flip_fluid
SOURCES = flip_fluid.cpp glad/src/glad.c

# ImGui sources
IMGUI_SOURCES = imgui/imgui.cpp \
                imgui/imgui_demo.cpp \
                imgui/imgui_draw.cpp \
                imgui/imgui_tables.cpp \
                imgui/imgui_widgets.cpp \
                imgui/backends/imgui_impl_glfw.cpp \
                imgui/backends/imgui_impl_opengl3.cpp

# Include directories
INCLUDES = -Iglad/include -Iimgui -Iimgui/backends

# Libraries
LIBS = -lglfw -ldl -lGL -lpthread

# Platform detection
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S), Darwin)
    LIBS = -lglfw -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo
endif

all: $(TARGET)

$(TARGET): $(SOURCES) $(IMGUI_SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCES) $(IMGUI_SOURCES) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean

# Usage:
# make          - Build the executable
# make clean    - Remove the executable
