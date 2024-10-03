#pragma once
#include <SDL.h>
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <SDL_opengles2.h>
#else
#include <SDL_opengl.h>
#endif
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_opengl3.h"
#include <vector>
void  InitTestImage();
bool LoadTextureFromRGBStream(GLuint texture, void *buffer, size_t width, size_t height);
void PlotTestImage();