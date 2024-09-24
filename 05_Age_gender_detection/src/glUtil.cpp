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
 bool LoadTextureFromColorStream(Inference_instance &stream, GLuint & texture) 
 {
    if(stream.pendingFrameCount == 0)
        return false;
   //std::cout << "OpenCV matrix size, Height:  " << stream.openGLfb.size().height << " Width: " << stream.openGLfb.size().width << std::endl;
    if(!stream.openGLfb.empty())
    {
        glBindTexture(GL_TEXTURE_2D, texture);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, stream.openGLfb.size().width, stream.openGLfb.size().height, 0, GL_RGB, GL_UNSIGNED_BYTE, stream.openGLfb.ptr());

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
bool FinishLoadTextureFromColorStream(Inference_instance &stream)
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