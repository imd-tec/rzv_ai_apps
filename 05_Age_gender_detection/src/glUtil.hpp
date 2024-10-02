 #pragma once
#include "inference.hpp"
#include <SDL.h>
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <SDL_opengles2.h>
#else
#include <SDL_opengl.h>
#endif
// BGR Texture binding
// Not used
GLuint CreateShaderProgram();

void Plot_And_Record_Stream_With_Custom_Shader(Inference_instance &handle, GLuint &texture, bool record, std::string windowName, 
    GLuint *shaderProgram);

// RGB
bool InitRGBTexture(Inference_instance &stream);
bool LoadTextureFromRGBStream(Inference_instance &stream) ;
bool FinishLoadTextureFromRGBStream(Inference_instance &stream);
void Plot_And_Record_Stream(Inference_instance &handle, GLuint &texture, bool record,std::string windowName);