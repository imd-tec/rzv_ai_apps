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

// Initialize shaders
GLuint loadShader(GLenum type, const char *shaderSrc) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &shaderSrc, NULL);
    glCompileShader(shader);

    GLint compiled;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        std::cout << "Failed to compile shader " << std::endl;
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

// Create and link program
GLuint createProgram(const char *vertexSource, const char *fragmentSource) {
    GLuint vertexShader = loadShader(GL_VERTEX_SHADER, vertexSource);
    GLuint fragmentShader = loadShader(GL_FRAGMENT_SHADER, fragmentSource);

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint linked;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    std::cout << "Programmed completly: " << linked << std::endl;
    if (!linked) {
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

// Function to plot an 8-bit BGR image using OpenGL ES
void BindBGRTexture(Inference_instance &stream) {
    // Vertex data for a full-screen quad
    GLfloat vertices[] = {
        -1.0f, -1.0f, 0.0f, 0.0f, 1.0f,  // Bottom left
         1.0f, -1.0f, 0.0f, 1.0f, 1.0f,  // Bottom right
        -1.0f,  1.0f, 0.0f, 0.0f, 0.0f,  // Top left
         1.0f,  1.0f, 0.0f, 1.0f, 0.0f   // Top right
    };

    // Load and compile shaders
    const char *vertexSource = "attribute vec4 aPosition; attribute vec2 aTexCoord; varying vec2 vTexCoord; void main() { gl_Position = aPosition; vTexCoord = aTexCoord; }";
    const char *fragmentSource = "precision mediump float; uniform sampler2D uTexture; varying vec2 vTexCoord; void main() { vec3 bgr = texture2D(uTexture, vTexCoord).bgr; gl_FragColor = vec4(bgr.b, bgr.g, bgr.r, 1.0); }";

    stream.program = createProgram(vertexSource, fragmentSource);
    glUseProgram(stream.program);

    // Vertex attributes
    stream.posAttrib = glGetAttribLocation(stream.program, "aPosition");
    stream.texAttrib = glGetAttribLocation(stream.program, "aTexCoord");
    glEnableVertexAttribArray(stream.posAttrib);
    glEnableVertexAttribArray(stream.texAttrib);

    // Specify vertices
    glVertexAttribPointer(stream.posAttrib, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), vertices);
    glVertexAttribPointer(stream.texAttrib, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), vertices + 3);

    glGenTextures(1, &stream.texture);
  
}


bool LoadTextureFromBGRStream(Inference_instance &stream)
{
    if(stream.pendingFrameCount == 0)
        return false;
   //std::cout << "OpenCV matrix size, Height:  " << stream.openGLfb.size().height << " Width: " << stream.openGLfb.size().width << std::endl;
    if(!stream.openGLfb.empty())
    {
        glUseProgram(stream.program);
        glActiveTexture(GL_TEXTURE0); // Activate texture unit 0
        glBindTexture(GL_TEXTURE_2D, stream.texture);
        // Upload the BGR data to the texture
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, stream.openGLfb.size().width, stream.openGLfb.size().height, 0, GL_RGB, GL_UNSIGNED_BYTE, stream.openGLfb.ptr());

        // Set texture filtering (optional, depending on the visual quality you want)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // Bind the texture to the shader
        glUniform1i(glGetUniformLocation(stream.program, "uTexture"), 0);

        // // Draw the quad
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glBindTexture(GL_TEXTURE_2D, 0); // Unbind texture when done
        stream.pendingFrameCount = 0; // Clear pending frame count
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
    if(stream.pendingFrameCount == 0)
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
        stream.pendingFrameCount = 0; // Clear pending frame count
        return true;
    }
    return false;

}
bool FinishLoadTextureFromRGBStream(Inference_instance &stream)
{
    /// TODO
}
void Plot_And_Record_Stream(Inference_instance &handle, GLuint &texture, bool record)
{
    if (handle.frameCounter > 0)
    {   
        if(handle.name.empty())
        {
            handle.name = "Unknown Stream";
        }
        ImGui::Begin(handle.name.c_str());
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