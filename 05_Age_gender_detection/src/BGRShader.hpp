#pragma once
#include <SDL2/SDL.h>
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <SDL2/SDL_opengles2.h>
#else
#include <SDL2/SDL_opengl.h>
#endif "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_opengl3.h"
#include <vector>
void  InitTestImage();
bool LoadTextureFromRGBStream(GLuint texture, void *buffer, size_t width, size_t height);
void PlotTestImage();