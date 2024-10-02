#include "inference.hpp"
#include <SDL.h>
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <SDL_opengles2.h>
#else
#include <SDL_opengl.h>
#endif
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_opengl3.h"
#include <GLES2/gl2.h>          // Use GL ES 2
const char* vertexShaderSource = R"(
#version 300 es  // Use OpenGL ES 3.0 version
precision mediump float;  // Define the default precision

layout(location = 0) in vec2 aPos;     // Vertex position
layout(location = 1) in vec2 aTexCoord; // Texture coordinate

out vec2 TexCoord; // Output to the fragment shader

void main()
{
    gl_Position = vec4(aPos, 0.0, 1.0); // Set the position of the vertex
    TexCoord = aTexCoord;               // Pass texture coordinate to fragment shader
}
)";

const char* fragmentShaderSource = R"(
#version 300 es  // Use OpenGL ES 3.0 version
precision mediump float;  // Define the default precision

in vec2 TexCoord;      // Input from vertex shader
out vec4 FragColor;    // Output color of the fragment

uniform sampler2D texture1; // The texture sampler

void main()
{
    vec4 color = texture(texture1, TexCoord); // Sample the texture
    // Swap the B and R channels to convert from BGR to RGB
    FragColor = vec4(color.b, color.g, color.r, color.a);
}
)";



GLuint CompileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        printf("ERROR::SHADER::COMPILATION_FAILED\n%s\n", infoLog);
    }
    return shader;
}
std::string LoadShaderSource(const char* filePath)
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

GLuint CreateShaderProgram() {

    std::string vertexStr = LoadShaderSource("vertex_shader.glsl");
    std::string fragStr = LoadShaderSource("fragment_shader.glsl");
    GLuint vertexShader = CompileShader(GL_VERTEX_SHADER, vertexStr.c_str());
    GLuint fragmentShader = CompileShader(GL_FRAGMENT_SHADER, fragStr.c_str());

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    GLint success;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        printf("ERROR::SHADER::PROGRAM::LINKING_FAILED\n%s\n", infoLog);
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    std::cout << "Created shader program with value " << shaderProgram << std::endl;
    return shaderProgram;
}




bool LoadTextureFromBGRStream(Inference_instance &stream)
{
    if (stream.pendingFrameCount == 0)
        return false;
   //std::cout << "OpenCV matrix size, Height:  " << stream.openGLfb.size().height << " Width: " << stream.openGLfb.size().width << std::endl;
    if(!stream.openGLfb.empty())
    {
        glBindTexture(GL_TEXTURE_2D, stream.texture);

        glTexImage2D(GL_TEXTURE_2D, 0,  GL_RGB, stream.openGLfb.size().width, stream.openGLfb.size().height, 0, GL_RGB, GL_UNSIGNED_BYTE, stream.openGLfb.ptr());

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0); // Unbind texture when done
        stream.pendingFrameCount = 0;
        return true;
    }
    return false;
}

bool InitRGBTexture(Inference_instance &stream)
{
    glGenTextures(1, &stream.texture);
}


 bool LoadTextureFromRGBStream(Inference_instance &stream) 
 {
    if (stream.pendingFrameCount == 0)
        return false;
   //std::cout << "OpenCV matrix size, Height:  " << stream.openGLfb.size().height << " Width: " << stream.openGLfb.size().width << std::endl;
    if(!stream.openGLfb.empty())
    {
        glBindTexture(GL_TEXTURE_2D, stream.texture);

        glTexImage2D(GL_TEXTURE_2D, 0,  GL_RGB, stream.openGLfb.size().width, stream.openGLfb.size().height, 0, GL_RGB, GL_UNSIGNED_BYTE, stream.openGLfb.ptr());

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0); // Unbind texture when done
        stream.pendingFrameCount = 0;
        return true;
    }
    return false;

}
bool FinishLoadTextureFromRGBStream(Inference_instance &stream)
{
    /// TODO
}
void Plot_And_Record_Stream(Inference_instance &handle, GLuint &texture, bool record, std::string windowName)
{
    if (handle.frameCounter > 0)
    {   
        if(handle.name.empty())
        {
            handle.name = "Unknown Stream";
        }
        ImGui::Begin(windowName.c_str());
        ImVec2 available_size = ImGui::GetContentRegionAvail();

        // Calculate the aspect ratio of the image
        const int32_t width = handle.openGLfb.size().width;
        const int32_t height = handle.openGLfb.size().height;
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
        ImGui::Image((void *)(intptr_t)texture, ImVec2(display_width, display_height));
        ImGui::End();
        #ifdef ENABLE_STATS
        std::string stats_window_name = handle.stream_name + " Statistics";
        ImGui::Begin(stats_window_name.c_str());
        if (handle.fps_values.size() > 50)
        {
            if (ImPlot::BeginPlot("Frame Statistics"))
            {
                ImPlot::SetupAxis(ImAxis_X1, "50 most recent frames");
                ImPlot::SetupAxis(ImAxis_Y1, "FPS (Hz)");
                uint32_t numPlots = 50;

                uint32_t start_index = handle.fps_values.size() - numPlots - 1;
                ImPlot::PlotLine(handle.stream_name.c_str(), &handle.fps_values[start_index], numPlots);
                ImPlot::EndPlot();
            }
        }

        ImGui::End();
        #endif
    }
}
void MyCustomShaderCallback(const ImDrawList* parent_list, const ImDrawCmd* cmd)
{
    //std::cout << "Using custom shader " << std::endl;
    // Use the custom shader program
    GLuint customShaderProgram = *(GLuint*)cmd->UserCallbackData;
    std::cout << "Using customer shader " << customShaderProgram;
        glUseProgram(customShaderProgram);
    //std::cout << "Finished using customer shader" << std::endl;
}

void ResetShaderCallback(const ImDrawList* parent_list, const ImDrawCmd* cmd)
{
    // Reset OpenGL state to ImGui's default
    glUseProgram(0);  // This disables the custom shader
}

void Plot_And_Record_Stream_With_Custom_Shader(Inference_instance &handle, GLuint &texture, bool record, std::string windowName, 
    GLuint *shaderProgram)
{
    if (handle.frameCounter > 0)
    {   
        if(handle.name.empty())
        {
            handle.name = "Unknown Stream";
        }
        ImGui::Begin(windowName.c_str());
        ImVec2 available_size = ImGui::GetContentRegionAvail();

        // Calculate the aspect ratio of the image
        const int32_t width = handle.openGLfb.size().width;
        const int32_t height = handle.openGLfb.size().height;
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
        ImVec2 p = ImGui::GetCursorScreenPos();  // Get the position of the image
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        // Add a callback to switch to the BGR shader
        draw_list->AddCallback(MyCustomShaderCallback, (void *)shaderProgram);
        // Draw the texture using ImGui, but it will use the custom shader due to the callback
        draw_list->AddImage((void *)(intptr_t)texture, p, ImVec2(p.x + display_width, p.y + display_height));
        //ImGui::Image((void *)(intptr_t)texture, ImVec2(display_width, display_height));

        // Add a callback to reset the shader back to default
        draw_list->AddCallback(ResetShaderCallback, nullptr);
        ImGui::End();
        #ifdef ENABLE_STATS
        std::string stats_window_name = handle.stream_name + " Statistics";
        ImGui::Begin(stats_window_name.c_str());
        if (handle.fps_values.size() > 50)
        {
            if (ImPlot::BeginPlot("Frame Statistics"))
            {
                ImPlot::SetupAxis(ImAxis_X1, "50 most recent frames");
                ImPlot::SetupAxis(ImAxis_Y1, "FPS (Hz)");
                uint32_t numPlots = 50;

                uint32_t start_index = handle.fps_values.size() - numPlots - 1;
                ImPlot::PlotLine(handle.stream_name.c_str(), &handle.fps_values[start_index], numPlots);
                ImPlot::EndPlot();
            }
        }

        ImGui::End();
        #endif
    }
}

