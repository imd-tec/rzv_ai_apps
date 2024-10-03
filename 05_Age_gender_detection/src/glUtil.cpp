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

GLuint c;
GLuint shaderProgram;

GLuint CompileShader(const char* source, GLenum shaderType)
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
std::string vertexShaderString = LoadShaderSource("vertex_shader.glsl");
std::string fragmentShaderString = LoadShaderSource("fragment_shader.glsl");
GLuint CreateShaderProgram()
{
    std::cout << "Creating shader program" << std::endl;

    GLuint vertexShader = CompileShader(vertexShaderString.c_str(), GL_VERTEX_SHADER);
    GLuint fragmentShader = CompileShader(fragmentShaderString.c_str(), GL_FRAGMENT_SHADER);

    shaderProgram = glCreateProgram();
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
    std::cout << "Loaded shader program" << std::endl;
    return shaderProgram;

}

bool LoadTextureFromBGRStream(Inference_instance &stream)
{
    if (stream.pendingFrameCount == 0)
        return false;
   //std::cout << "OpenCV matrix size, Height:  " << stream.openGLfb.size().height << " Width: " << stream.openGLfb.size().width << std::endl;
    if(!stream.openGLfb->fb.empty())
    {
        glBindTexture(GL_TEXTURE_2D, stream.texture);

        glTexImage2D(GL_TEXTURE_2D, 0,  GL_RGB, stream.openGLfb->fb.size().width, stream.openGLfb->fb.size().height, 0, GL_RGB, GL_UNSIGNED_BYTE, stream.openGLfb->fb.ptr());

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

GLuint InitCustomShaderProgram()
{
    return CreateShaderProgram();
}

 bool LoadTextureFromRGBStream(Inference_instance &stream) 
 {
    if (stream.pendingFrameCount == 0)
        return false;
   //std::cout << "OpenCV matrix size, Height:  " << stream.openGLfb.size().height << " Width: " << stream.openGLfb.size().width << std::endl;
    if(!stream.openGLfb->fb.empty())
    {
        // Faster to copy the buffer to cached memory than use the uncached memory mapped buffers
        auto start = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        //cv::Mat test(cv::Size(stream.openGLfb->fb.size().width, stream.openGLfb->fb.size().height), CV_8UC3);

         
        //test = stream.openGLfb->fb.clone();
      
       
        glBindTexture(GL_TEXTURE_2D, stream.texture);

        glTexImage2D(GL_TEXTURE_2D, 0,  GL_RGB, stream.openGLfb->fb.size().width, stream.openGLfb->fb.size().height, 0, GL_RGB, GL_UNSIGNED_BYTE, stream.openGLfb->fb.ptr());

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0); // Unbind texture when done
        auto end = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        stream.pendingFrameCount = 0;

        
        std::cout << "Bind texture time " << end-start << std::endl;
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
        // ImGui::Begin(windowName.c_str());
        // ImVec2 size(1000.0f, 600.0f);
        // ImGui::InvisibleButton("canvas", size);
        // ImVec2 available_size = ImGui::GetContentRegionAvail();

        // // Calculate the aspect ratio of the image
        // const int32_t width = handle.openGLfb.size().width;
        // const int32_t height = handle.openGLfb.size().height;
        // float aspect_ratio = (float)width / (float)height;

        // // Determine the appropriate width and height to maintain aspect ratio
        // float display_width = available_size.x;
        // float display_height = available_size.y;

        // if (display_width / aspect_ratio <= display_height)
        // {
        //     display_height = display_width / aspect_ratio;
        // }
        // else
        // {
        //     display_width = display_height * aspect_ratio;
        // }
        // //std::cout << "Plotting with width : " << display_width << " and height: " << display_height << std::endl;
        // ImGui::Image((void *)(intptr_t)texture, ImVec2(display_width, display_height));
        // ImGui::End();
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
    //glBindTexture(GL_TEXTURE_2D, 0); // Unbind the texture
    glUseProgram(shaderProgram);                 // Unbind any custom shader
    GLuint texture = *(GLuint * ) cmd->UserCallback;

    //glBindTexture(GL_TEXTURE_2D, texture);
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
        // Set the sampler uniform to use texture unit 0
    glUniform1i(glGetUniformLocation(shaderProgram, "Texture"), 0);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "ProjMtx"), 1, GL_FALSE, &ortho_projection[0][0]);    
    glDrawElements(GL_TRIANGLES, (GLsizei)cmd->ElemCount, sizeof(ImDrawIdx) == 2 ? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT, (void*)(intptr_t)(cmd->IdxOffset * sizeof(ImDrawIdx)));
}

void ResetShaderCallback(const ImDrawList* parent_list, const ImDrawCmd* cmd)
{
    // Reset OpenGL state to ImGui's default
    glUseProgram(0);  // This disables the custom shader
}

void Plot_And_Record_Stream_With_Custom_Shader(Inference_instance &handle, GLuint &texture, bool record, std::string windowName)
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
        const int32_t width = handle.openGLfb->fb.size().width;
        const int32_t height = handle.openGLfb->fb.size().height;
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
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        // Add a callback to switch to the BGR shader

        //draw_list->PushClipRect(p0, p1);
        draw_list->AddCallback(MyCustomShaderCallback, &handle.texture);
        // draw_list->AddRectFilled(p0, p1, 0xFFFF00FF);
        // draw_list->PopClipRect();
        // Draw the texture using ImGui, but it will use the custom shader due to the callback
        //draw_list->PushClipRect(p.x, p.y);
        //draw_list->AddImage((void *)(intptr_t)texture, p, ImVec2(p.x + display_width, p.y + display_height));
        ImGui::Image((void *)(intptr_t)texture, ImVec2(display_width, display_height));

        // Add a callback to reset the shader back to default
        draw_list->AddCallback(ImDrawCallback_ResetRenderState, nullptr);
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

