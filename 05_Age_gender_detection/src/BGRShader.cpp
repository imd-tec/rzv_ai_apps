#define GL_GLEXT_PROTOTYPES
#include <SDL2/SDL.h>
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <SDL2/SDL_opengles2.h>
#else
#include <SDL2/SDL_opengl.h>
#endif
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_opengl3.h"
#include <vector>
#include <ostream>
#include <iostream>
#include <string>
#include <ostream>
#include <GLES2/gl2.h>          // Use GL ES 2
#include <fstream>
#include <sstream>
const GLchar* vertex_shader_glsl_410_core =
        "#version 410\n"
        "layout (location = 0) in vec2 Position;\n"
        "layout (location = 1) in vec2 UV;\n"
        "layout (location = 2) in vec4 Color;\n"
        "uniform mat4 ProjMtx;\n"
        "out vec2 Frag_UV;\n"
        "out vec4 Frag_Color;\n"
        "void main()\n"
        "{\n"
        "    Frag_UV = UV;\n"
        "    Frag_Color = Color;\n"
        "    gl_Position = ProjMtx * vec4(Position.xy,0,1);\n"
        "}\n";
const GLchar* fragment_shader_glsl_410_core =
        "#version 410\n"
        "in vec2 Frag_UV;\n"
        "in vec4 Frag_Color;\n"
        "uniform sampler2D Texture;\n"
        "layout (location = 0) out vec4 Out_Color;\n"
        "void main()\n"
        "{\n"
        "    Out_Color = Frag_Color * texture(Texture, Frag_UV.st);\n"
        "}\n";
static std::vector<uint8_t> testImage = std::vector<uint8_t>();
GLuint texture;
GLuint custom_BGR_Shader;

GLuint CustomCompileShader(const char* source, GLenum shaderType)
{
    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    // Check for compilation errors
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Error: Shader Compilation Failed\n" << infoLog << std::endl;
    }

    return shader;
}
std::string CustomShaderSource(const char* filePath)
{
    std::ifstream shaderFile(filePath);
    std::stringstream shaderStream;

    if (!shaderFile.is_open())
    {
        std::cerr << "Could not open file: " << filePath << std::endl;
        return "";
    }

    shaderStream << shaderFile.rdbuf(); // Read the file buffer into the stream
    shaderFile.close();

    return shaderStream.str(); // Return the string containing shader source
}
GLuint CustomCreateShaderProgram()
{
    std::string vertexShaderString = CustomShaderSource("vertex_shader.glsl");
    std::string fragmentShaderString = CustomShaderSource("fragment_shader.glsl");

    GLuint vertexShader = CustomCompileShader(vertexShaderString.c_str(), GL_VERTEX_SHADER);
    GLuint fragmentShader = CustomCompileShader(fragmentShaderString.c_str(), GL_FRAGMENT_SHADER);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Check for linking errors
    GLint success;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        std::cerr << "Error: Shader Program Linking Failed\n" << infoLog << std::endl;
    }

    // Delete shaders after linking
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}
void InitTestImage()
{
    for(int i = 0; i < 1080*1920; i ++)
    {
        testImage.push_back(128);
        testImage.push_back(100);
        testImage.push_back(55);
    }
    glGenTextures(1, &texture); 
    custom_BGR_Shader = CustomCreateShaderProgram();
}

bool LoadTextureFromRGBStream(GLuint texture, void *buffer, size_t width, size_t height)
 {

   //std::cout << "OpenCV matrix size, Height:  " << stream.openGLfb.size().height << " Width: " << stream.openGLfb.size().width << std::endl;

    {
        glBindTexture(GL_TEXTURE_2D, texture);

        glTexImage2D(GL_TEXTURE_2D, 0,  GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, buffer);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0); // Unbind texture when done

        return true;
    }
    return false;

}
void OurCustomCallback(const ImDrawList* parent_list, const ImDrawCmd* pcmd)
{
    //glBindTexture(GL_TEXTURE_2D, 0); // Unbind the texture
    glUseProgram(custom_BGR_Shader);                 // Unbind any custom shader

    glBindTexture(GL_TEXTURE_2D, texture);
    ImDrawData* draw_data = ImGui::GetDrawData();
    float L = draw_data->DisplayPos.x;
    float R = draw_data->DisplayPos.x + draw_data->DisplaySize.x;
    float T = draw_data->DisplayPos.y;
    float B = draw_data->DisplayPos.y + draw_data->DisplaySize.y;

    const float ortho_projection[4][4] =
    {
        { 2.0f / (R - L),   0.0f,         0.0f,   0.0f },
        { 0.0f,         2.0f / (T - B),   0.0f,   0.0f },
        { 0.0f,         0.0f,        -1.0f,   0.0f },
        { (R + L) / (L - R),  (T + B) / (B - T),  0.0f,   1.0f },
    };
    glUniformMatrix4fv(glGetUniformLocation(custom_BGR_Shader, "ProjMtx"), 1, GL_FALSE, &ortho_projection[0][0]);
    //glDrawElements(GL_TRIANGLES, (GLsizei)pcmd->ElemCount, sizeof(ImDrawIdx) == 2 ? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT, (void*)(intptr_t)(pcmd->IdxOffset * sizeof(ImDrawIdx)));
}
void CustomImagePlot(ImTextureID user_texture_id, const ImVec2& image_size, const ImVec4& tint_col)
{

    ImDrawList*  draw_list = ImGui::GetWindowDrawList();
    // Render
    auto pMin = ImVec2(0,0);
    auto pMax = ImVec2(1920,1080);

    ImVec2 uv0 = ImVec2(-1,-1);
    ImVec2 uv1 = ImVec2(0.5,0.5);
    draw_list->AddCallback(OurCustomCallback, (void *)NULL);    
    draw_list->AddImage(user_texture_id, pMin, pMax, uv0, uv1, ImGui::GetColorU32(tint_col));
    //draw_list->AddCallback(ImDrawCallback_ResetRenderState, nullptr);
}

void PlotTestImage()
{

    LoadTextureFromRGBStream(texture,testImage.data(),1920,1080);
    ImGui::Begin("Test");
    


        ImVec2 available_size = ImGui::GetContentRegionAvail();
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        // Calculate the aspect ratio of the image
        const int32_t width = 1920;
        const int32_t height =  1080;
        float aspect_ratio = (float)width / (float)height;

        // Determine the appropriate width and height to maintain aspect ratio
        float display_width = available_size.x;
        float display_height = available_size.y;

        if (display_width / aspect_ratio <= display_height)
        {
            display_height = display_width / aspect_ratio;
        }
        else
        {
            display_width = display_height * aspect_ratio;
        }
        //std::cout << "Plotting with width : " << display_width << " and height: " << display_height << std::endl;
        auto tint = ImVec4(1, 1, 1, 1);
        draw_list->AddCallback(OurCustomCallback, (void *)NULL);  
        ImGui::Image((void *)(intptr_t)texture, ImVec2(display_width, display_height));
  
        draw_list->AddCallback(ImDrawCallback_ResetRenderState, nullptr);
        ImGui::End();
}


