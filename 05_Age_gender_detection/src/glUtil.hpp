 #pragma once
#include "inference.hpp"
#include <SDL.h>
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <SDL_opengles2.h>
#else
#include <SDL_opengl.h>
#endif

 
 
 bool LoadTextureFromColorStream(Inference_instance &stream, GLuint & texture) ;
 bool FinishLoadTextureFromColorStream(Inference_instance &stream);
 void Plot_And_Record_Stream(Inference_instance &handle, GLuint &texture, bool record);