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
#include <deque>
#include <condition_variable>
		
// Object for capturing from a camera
struct Inference_instance
{
    std::string gstreamer_pipeline;
    std::string device;
    uint32_t index;
    std::string name = "Instance";
    std::string age[NUM_MAX_FACE];
    std::string gender[NUM_MAX_FACE];
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
    uint64_t headTimestamp = 0;
    uint32_t lastHeadCount = 0;
    // Thread for processing frames after reading from GStreamer
    std::condition_variable frameProcessThreadWakeup;
    std::deque<std::shared_ptr<V4L_ZeroCopyFB>>     frameProcessQ = std::deque<std::shared_ptr<V4L_ZeroCopyFB>>();
    std::thread            frameProcessThread;
    std::mutex              frameProcessMutex = std::mutex();
    // Face detect theead for processing frames
    std::thread            faceDetectThread;
    std::condition_variable faceDetectWakeUp;
    std::deque<cv::Mat>            faceDetectQ = std::deque<cv::Mat> ();
    std::mutex              faceDetectMutex = std::mutex();
    // Results mutex 
    std::mutex              faceDetectResultsMutex = std::mutex();
    // Statistics
    std::chrono::microseconds infTimeTinyFace;
    std::mutex timestampMtx;
    std::list<std::chrono::system_clock::time_point> Frame_Timestamp =  std::list<std::chrono::system_clock::time_point>();
    std::chrono::system_clock::time_point previousTimestamp;
};

struct Inference_Statistics
{
    std::list<std::chrono::microseconds> Yolo_preInferenceTime = std::list<std::chrono::microseconds>(); 
    std::list<std::chrono::microseconds> Yolo_inferenceTime = std::list<std::chrono::microseconds>();
    std::list<std::chrono::microseconds> Yolo_postInferenceTime =  std::list<std::chrono::microseconds>();
    std::list<std::chrono::microseconds> FairFace_preInferenceTime = std::list<std::chrono::microseconds>(); 
    std::list<std::chrono::microseconds> FairFace_inferenceTime = std::list<std::chrono::microseconds>();
    std::list<std::chrono::microseconds> FairFace_postInferenceTime =  std::list<std::chrono::microseconds>();
    
    std::mutex mtx;
};