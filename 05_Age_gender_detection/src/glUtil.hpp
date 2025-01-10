 #pragma once
#include "inference.hpp"
#include <SDL2/SDL.h>
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <SDL2/SDL_opengles2.h>
#else
#include <SDL2/SDL_opengl.h>
#endif
// BGR Texture binding
// Not used
void CreateShaderProgram();

void Plot_And_Record_Stream_With_Custom_Shader(Inference_instance &handle, GLuint &texture, bool record, std::string windowName);

// RGB
bool InitRGBTexture(Inference_instance &stream);
bool LoadTextureFromRGBStream(Inference_instance &stream) ;
bool FinishLoadTextureFromRGBStream(Inference_instance &stream);
void Plot_And_Record_Stream(Inference_instance &handle, GLuint &texture, bool record,std::string windowName);
GLuint InitCustomShaderProgram();

// Logo 
cv::Mat LoadLogoTexture(std::string filePath);
void BindLogoTexture(cv::Mat logoTexture, GLuint &logo_texture);
void PlotLogoImage( GLuint &logo_texture);