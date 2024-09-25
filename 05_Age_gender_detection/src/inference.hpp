#pragma once
#include <string>
#include <stdint.h>
#include "define.h"
#include <opencv2/opencv.hpp>
#include "inference.hpp"
#include <SDL.h>
#include <thread>
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <SDL_opengles2.h>
#else
#include <SDL_opengl.h>
#endif
#include "v4lutil.hpp"

// Object for capturing from a camera
struct Inference_instance
{
    std::string gstreamer_pipeline;
    std::string device;
    uint32_t index;
    std::string name = "Instance";
    std::string age;
    std::string gender;
    int16_t cropx1[NUM_MAX_FACE];
    int16_t cropy1[NUM_MAX_FACE];
    int16_t cropx2[NUM_MAX_FACE];
    int16_t cropy2[NUM_MAX_FACE];
    cv::Mat g_frame;
    cv::Mat openGLfb;
    std::mutex openGLfbMutex = std::mutex();
    cv::VideoCapture cap;
    uint32_t DisplayStartX = 0;
    uint32_t DisplayStartY = 0;
    GLuint texture;                 // This is going to be recycled
    uint32_t frameCounter = 0 ;     // Total number of frames
    uint32_t pendingFrameCount = 0;   // Pending frames to process
    std::shared_ptr<V4LUtil> v4lUtil; // Capture instance
    GLuint program;
    GLuint posAttrib;
    GLuint texAttrib;
    uint32_t headCount = 0;
};