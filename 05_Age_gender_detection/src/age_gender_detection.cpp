/*
 * Original Code (C) Copyright Edgecortix, Inc. 2022
 * Modified Code (C) Copyright Renesas Electronics Corporation 2023
 *　
 *  *1 DRP-AI TVM is powered by EdgeCortix MERA(TM) Compiler Framework.
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an  
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 */

/***********************************************************************************************************************
* File Name    : Age_gender_detection.cpp
* Version      : 1.1.0
* Description  : DRP-AI TVM[*1] Application Example
***********************************************************************************************************************/

/*****************************************
* includes
******************************************/
#include "define.h"
#include "box.h"
#include "MeraDrpRuntimeWrapper.h"
#include <linux/drpai.h>
#include <linux/input.h>
#include <builtin_fp16.h>
#include <opencv2/opencv.hpp>
#include "wayland.h"
#include <mutex>
#include <thread>
#include <condition_variable>
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>
#include <SDL.h>
#include "inference.hpp"
#include "glUtil.hpp"
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <SDL_opengles2.h>
#else
#include <SDL_opengl.h>
#endif
#include "statisticsPlot.hpp"
// This example can also compile and run with Emscripten! See 'Makefile.emscripten' for details.
#ifdef __EMSCRIPTEN__
#include "../libs/emscripten/emscripten_mainloop_stub.h"
#endif
#include "implot.h"
#include "BGRShader.hpp"

using namespace std;
using namespace cv;

/* DRP-AI TVM[*1] Runtime object */
MeraDrpRuntimeWrapper runtime;
MeraDrpRuntimeWrapper runtime1;

static Wayland wayland;
static pthread_t ai_inf_thread;
static pthread_t kbhit_thread;
static sem_t terminate_req_sem;
static int32_t drpai_freq;

// Main loop
static bool thread_done = false;

/*Global Variables*/
static float drpai_output_buf[INF_OUT_SIZE_TINYYOLOV2];
static float drpai_output_buf1[INF_OUT_SIZE_FAIRFACE];

static atomic<uint8_t> hdmi_obj_ready   (0);

static uint32_t disp_time = 0;

std::string media_port;
std::string gstreamer_pipeline;
std::string gstreamerSecond;


std::vector<float> floatarr(1);

uint64_t drpaimem_addr_start = 0;
bool runtime_status = false; 
bool runtime_status1 = false; 


std::atomic<uint64_t> timestamp_detection = 0;
int fd;

float POST_PROC_TIME_TINYYOLO =0;
float POST_PROC_TIME_FACE =0;
float PRE_PROC_TIME_TINYYOLO =0;
float PRE_PROC_TIME_FACE =0;
float INF_TIME_FACE = 0;
float INF_TIME_TINYYOLO = 0;

Inference_instance instances[2];
static std::string age_range[9] = {"0-2", "3-9","10-19","20-29","30-39","40-49","50-59","60-69","70+"} ;
static std::string gender_ls[2] = {"Male", "Female"};
/*Inference mutex*/
std::mutex drpAIMutex; // Mutex to protect the DRP engine
#define NUM_FRAME_BUFFERS 20
/*Global frame */
// Used for OpenCV calls
// RGBA buffer passed to wayland
std::array<std::vector<uint8_t>,NUM_FRAME_BUFFERS> output_fb;

// Protects output_image and output_fb
std::mutex output_mutex;
int instanceIndex = 0; // Which instance to run face detect on
int output_fb_ready[2]  = {0,0};
int output_fb_index = 0;

/* Map to store input source list */
static int input_source = INPUT_SOURCE_USB;
std::map<std::string, int> input_source_map ={    
    {"USB", INPUT_SOURCE_USB},
    {"MIPI", INPUT_SOURCE_MIPI}
    } ;

Inference_Statistics inferenceStatistics;
/*****************************************
 * Function Name     : float16_to_float32
 * Description       : Function by Edgecortex. Cast uint16_t a into float value.
 * Arguments         : a = uint16_t number
 * Return value      : float = float32 number
 ******************************************/
float float16_to_float32(uint16_t a)
{
    return __extendXfYf2__<uint16_t, uint16_t, 10, float, uint32_t, 23>(a);
}

/*****************************************
 * Function Name     : load_label_file
 * Description       : Load label list text file and return the label list that contains the label.
 * Arguments         : label_file_name = filename of label list. must be in txt format
 * Return value      : vector<string> list = list contains labels
 *                     empty if error occurred
 ******************************************/
vector<string> load_label_file(string label_file_name)
{
    vector<string> list = {};
    vector<string> empty = {};
    ifstream infile(label_file_name);

    if (!infile.is_open())
    {
        return list;
    }

    string line = "";
    while (getline(infile, line))
    {
        list.push_back(line);
        if (infile.fail())
        {
            return empty;
        }
    }

    return list;
}

/*****************************************
 * Function Name : sigmoid
 * Description   : Helper function for YOLO Post Processing
 * Arguments     : x = input argument for the calculation
 * Return value  : sigmoid result of input x
 ******************************************/
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

/*****************************************
* Function Name : softmax
* Description   : Helper function for YOLO Post Processing
* Arguments     : val[] = array to be computed softmax
* Return value  : -
******************************************/
static void softmax(float val[NUM_CLASS])
{
    float max_num = -FLT_MAX;
    float sum = 0;
    int32_t i;
    for ( i = 0 ; i<NUM_CLASS ; i++ )
    {
        max_num = std::max(max_num, val[i]);
    }

    for ( i = 0 ; i<NUM_CLASS ; i++ )
    {
        val[i]= (float) exp(val[i] - max_num);
        sum+= val[i];
    }

    for ( i = 0 ; i<NUM_CLASS ; i++ )
    {
        val[i]= val[i]/sum;
    }
    return;
}


/*****************************************
* Function Name : index
* Description   : Get the index of the bounding box attributes based on the input offset
* Arguments     : offs = offset to access the bounding box attributes
*                 channel = channel to access each bounding box attribute.
* Return value  : index to access the bounding box attribute.
******************************************/
static int32_t index(int32_t offs, int32_t channel)
{
    return offs + channel * NUM_GRID_X * NUM_GRID_Y;
}

/*****************************************
* Function Name : offset_yolo
* Description   : Get the offset number to access the bounding box attributes
*                 To get the actual value of bounding box attributes, use index() after this function.
* Arguments     : b = Number to indicate which bounding box in the region [0~4]
*                 y = Number to indicate which region [0~13]
*                 x = Number to indicate which region [0~13]
* Return value  : offset to access the bounding box attributes.
*******************************************/
static int offset_yolo(int b, int y, int x)
{
    return b *(NUM_CLASS + 5)* NUM_GRID_X * NUM_GRID_Y + y * NUM_GRID_X + x;
}

static int8_t wait_join(pthread_t *p_join_thread, uint32_t join_time)
{
    int8_t ret_err;
    struct timespec join_timeout;
    ret_err = clock_gettime(CLOCK_REALTIME, &join_timeout);
    if ( 0 == ret_err )
    {
        join_timeout.tv_sec += join_time;
        ret_err = pthread_timedjoin_np(*p_join_thread, NULL, &join_timeout);
    }
    return ret_err;
}


/*****************************************
* Function Name : R_Post_Proc_ResNet34
* Description   : CPU post-processing for Resnet34 
* Arguments     : floatarr = drpai output address
*                 n_pers = number of the face detected
* Return value  : -
******************************************/
static void R_Post_Proc_ResNet34(float* floatarr, uint8_t n_pers, Inference_instance &instance)
{
    float max = std::numeric_limits<float>::min();
    int8_t index = -1;
    for(int8_t i=9;i<INF_OUT_SIZE_FAIRFACE;i++)
    {
        if(floatarr[i]>max)
        {
            max = floatarr[i];
            index = i;
        }
    }
    
    instance.age[n_pers] = age_range[index-9];

    if (floatarr[7] > floatarr[8])
    {
        instance.gender[n_pers] = gender_ls[0];
    } 
    else
    {
        instance.gender[n_pers] =  gender_ls[1];
    }
    // Store timestamp and then clear the display after 2 seconds
    timestamp_detection =  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    return;
}


/*****************************************
 * Function Name : R_Post_Proc
 * Description   : Process CPU post-processing for YOLOv3
 * Arguments     : floatarr = drpai output address
 * Return value  : -
 ******************************************/
void R_Post_Proc(float *floatarr, Inference_instance &instance, std::vector<detection> &det)
{
    /* Following variables are required for correct_region_boxes in Darknet implementation*/
    /* Note: This implementation refers to the "darknet detector test" */
    std::scoped_lock resultslk(instance.faceDetectResultsMutex);
    float new_w, new_h;
    int32_t result_cnt =0;
    int32_t count =0;
    float correct_w = 1.;
    float correct_h = 1.;
    if ((float)(MODEL_IN_W / correct_w) < (float)(MODEL_IN_H / correct_h))
    {
        new_w = (float)MODEL_IN_W;
        new_h = correct_h * MODEL_IN_W / correct_w;
    }
    else
    {
        new_w = correct_w * MODEL_IN_H / correct_h;
        new_h = MODEL_IN_H;
    }

    int32_t b = 0;
    int32_t y = 0;
    int32_t x = 0;
    int32_t offs = 0;
    int32_t i = 0;
    float tx = 0;
    float ty = 0;
    float tw = 0;
    float th = 0;
    float tc = 0;
    float center_x = 0;
    float center_y = 0;
    float box_w = 0;
    float box_h = 0;
    float objectness = 0;
    Box bb;
    float classes[NUM_CLASS];
    float max_pred = 0;
    int32_t pred_class = -1;
    float probability = 0;
    detection d;
    det.clear();

    /*Post Processing Start*/
    for(b = 0; b < NUM_BB; b++)
    {
        for(y = 0; y < NUM_GRID_Y; y++)
        {
            for(x = 0; x < NUM_GRID_X; x++)
            {
                offs = offset_yolo(b, y, x);
                tx = floatarr[offs];
                ty = floatarr[index(offs, 1)];
                tw = floatarr[index(offs, 2)];
                th = floatarr[index(offs, 3)];
                tc = floatarr[index(offs, 4)];

                /* Compute the bounding box */
                /*get_region_box*/
                center_x = ((float) x + sigmoid(tx)) / (float) NUM_GRID_X;
                center_y = ((float) y + sigmoid(ty)) / (float) NUM_GRID_Y;
                box_w = (float) exp(tw) * anchors[2*b+0] / (float) NUM_GRID_X;
                box_h = (float) exp(th) * anchors[2*b+1] / (float) NUM_GRID_Y;

                /* Adjustment for VGA size */
                /* correct_region_boxes */
                center_x = (center_x - (MODEL_IN_W - new_w) / 2. / MODEL_IN_W) / ((float) new_w / MODEL_IN_W);
                center_y = (center_y - (MODEL_IN_H - new_h) / 2. / MODEL_IN_H) / ((float) new_h / MODEL_IN_H);
                box_w *= (float) (MODEL_IN_W / new_w);
                box_h *= (float) (MODEL_IN_H / new_h);

                center_x = round(center_x * DRPAI_IN_WIDTH);
                center_y = round(center_y * DRPAI_IN_HEIGHT);
                box_w = round(box_w * DRPAI_IN_WIDTH);
                box_h = round(box_h * DRPAI_IN_HEIGHT);

                objectness = sigmoid(tc);

                bb = {center_x, center_y, box_w, box_h};
                /* Get the class prediction */
                for (i = 0; i < NUM_CLASS; i++)
                {
                    classes[i] = floatarr[index(offs, 5+i)];
                }
                softmax(classes);
                max_pred = 0;
                pred_class = -1;
                for (i = 0; i < NUM_CLASS; i++)
                {
                    if (classes[i] > max_pred)
                    {
                        pred_class = i;
                        max_pred = classes[i];
                    }
                }

                /* Store the result into the list if the probability is more than the threshold */
                probability = max_pred * objectness;
                if ((probability > TH_PROB))
                {
              
                    d = {bb, pred_class, probability};
                    det.push_back(d);
                    
                }
            }
        }
    }
    /* Non-Maximum Supression filter */
    filter_boxes_nms(det, det.size(), TH_NMS);
    for (i = 0; i < det.size(); i++)
    {
        /* Skip the overlapped bounding boxes */
        if (det[i].prob == 0)
        {
            continue;
        }
        else{
            result_cnt++;
            if(count > 2)
            {
                break;
            }
        }
    }
    instance.headTimestamp =  std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count(); 
    instance.headCount = result_cnt++;
    if(instance.headCount)
        instance.lastHeadCount = instance.headCount;
    return;


    return;
}
void Push_Statistic(std::chrono::microseconds value, std::list<std::chrono::microseconds> &q)
{
    if(q.size() == MAX_STATISTIC_SIZE)
        q.pop_front();
    q.push_back(value);
}

/*****************************************
 * Function Name : Face Detection
 * Description   : Function to perform over all detection
 * Arguments     : -
 * Return value  : 0 if succeeded
 *               not 0 otherwise
 ******************************************/
int Face_Detection(cv::Mat inputFrame, Inference_instance &instance)
{   
    int wait_key;
     /* Temp frame */
    Mat frame1;

    Size size(MODEL_IN_H, MODEL_IN_W);
    /*Pre process start time for tinyyolo model */
    auto t0 = std::chrono::high_resolution_clock::now();
    /*resize the image to the model input size*/
    resize(inputFrame, frame1, size);

    // printf("frame1: d=%d c=%d rows=%d cols=%d\n", frame1.depth(), frame1.channels(), frame1.rows, frame1.cols);


    /* changing channel from hwc to chw */
    vector<Mat> rgb_images;
    split(frame1, rgb_images);
    Mat m_flat_r = rgb_images[0].reshape(1, 1);
    Mat m_flat_g = rgb_images[1].reshape(1, 1);
    Mat m_flat_b = rgb_images[2].reshape(1, 1);
    Mat matArray[] = {m_flat_r, m_flat_g, m_flat_b};
    Mat frameCHW;
    hconcat(matArray, 3, frameCHW);
    /*convert to FP32*/
    frameCHW.convertTo(frameCHW, CV_32FC3);

    /* normailising  pixels */
    divide(frameCHW, 255.0, frameCHW);

    /* DRP AI input image should be continuous buffer */
    if (!frameCHW.isContinuous())
        frameCHW = frameCHW.clone();

    Mat frame = frameCHW;
    int ret = 0;

    /* Preprocess time ends for tinyyolo model*/
    auto t1 = std::chrono::high_resolution_clock::now();
    
     /* tinyyolov2 inference*/
    /*start inference using drp runtime*/
    runtime.SetInput(0, frame.ptr<float>());
    
    /* Inference start time for tinyyolo model*/
    auto t2 = std::chrono::high_resolution_clock::now();
    runtime.Run(drpai_freq);
    /* Inference time end for tinyyolo model */
    auto t3 = std::chrono::high_resolution_clock::now();
    auto inf_duration = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    /* Postprocess time start for tinyyolo model */
    auto t4 = std::chrono::high_resolution_clock::now();
    /*load inference out on drpai_out_buffer*/
    int32_t i = 0;
    int32_t output_num = 0;
    std::tuple<InOutDataType, void *, int64_t> output_buffer;
    int64_t output_size;
    uint32_t size_count = 0;

    /* Get the number of output of the target model. */
    output_num = runtime.GetNumOutput();
    size_count = 0;
    /*GetOutput loop*/
    for (i = 0; i < output_num; i++)
    {
        /* output_buffer below is tuple, which is { data type, address of output data, number of elements } */
        output_buffer = runtime.GetOutput(i);
        /*Output Data Size = std::get<2>(output_buffer). */
        output_size = std::get<2>(output_buffer);

        /*Output Data Type = std::get<0>(output_buffer)*/
        if (InOutDataType::FLOAT16 == std::get<0>(output_buffer))
        {
            /*Output Data = std::get<1>(output_buffer)*/
            uint16_t *data_ptr = reinterpret_cast<uint16_t *>(std::get<1>(output_buffer));
            for (int j = 0; j < output_size; j++)
            {
                /*FP16 to FP32 conversion*/
                drpai_output_buf[j + size_count] = float16_to_float32(data_ptr[j]);
            }
        }
        else if (InOutDataType::FLOAT32 == std::get<0>(output_buffer))
        {
            /*Output Data = std::get<1>(output_buffer)*/
            float *data_ptr = reinterpret_cast<float *>(std::get<1>(output_buffer));
            for (int j = 0; j < output_size; j++)
            {
                drpai_output_buf[j + size_count] = data_ptr[j];
            }
        }
        else
        {
            std::cerr << "[ERROR] Output data type : not floating point." << std::endl;
            ret = -1;
            break;
        }
        size_count += output_size;
    }

    if (ret != 0)
    {
        std::cerr << "[ERROR] DRP Inference Not working !!! " << std::endl;
        return -1;
    }
    std::vector<detection> det = std::vector<detection>();
    det.clear();
   /* Do post process to get bounding boxes */
    R_Post_Proc(drpai_output_buf,instance,det );
    /*/* Postprocess time end for tinyyolo model*/
    auto t5 = std::chrono::high_resolution_clock::now();
    std::scoped_lock resultslk(instance.faceDetectResultsMutex);
    if (instance.headCount > 0){

       float POST_PROC_TIME_FACE_MICRO =0;
       float PRE_PROC_TIME_FACE_MICRO =0;
       float INF_TIME_FACE_MICRO = 0;
        for (int i = 0 ; i < instance.headCount; i++)
        {
        
            /* Preprocess time start for fairface model*/
            auto t0_face = std::chrono::high_resolution_clock::now();

            instance.cropx1[i] = (int)det[i].bbox.x - round((int)det[i].bbox.w / 2.);
            instance.cropy1[i] = (int)det[i].bbox.y - round((int)det[i].bbox.h / 2.);
            instance.cropx2[i] = (int)det[i].bbox.x + round((int)det[i].bbox.w / 2.) - 1;
            instance.cropy2[i] = (int)det[i].bbox.y + round((int)det[i].bbox.h / 2.) - 1;

            /* Check the bounding box is in the image range */
            instance.cropx1[i] = instance.cropx1[i] < 1 ? 1 : instance.cropx1[i];
            instance.cropx2[i] = ((DRPAI_IN_WIDTH - 2) < instance.cropx2[i]) ? (DRPAI_IN_WIDTH - 2) : instance.cropx2[i];
            instance.cropy1[i] = instance.cropy1[i] < 1 ? 1 : instance.cropy1[i];
            instance.cropy2[i] = ((DRPAI_IN_HEIGHT - 2) < instance.cropy2[i]) ? (DRPAI_IN_HEIGHT - 2) : instance.cropy2[i];
            Mat cropped_image = instance.g_frame(Range(instance.cropy1[i],instance.cropy2[i]), Range(instance.cropx1[i],instance.cropx2[i]));
            Mat frame1res;
            Size size(MODEL1_IN_H, MODEL1_IN_W);

            /*resize the image to the model input size*/
            resize(cropped_image, frame1res, size);
            vector<Mat> rgb_imagesres;
            split(frame1res, rgb_imagesres);
            Mat m_flat_r_res = rgb_imagesres[0].reshape(1, 1);
            Mat m_flat_g_res = rgb_imagesres[1].reshape(1, 1);
            Mat m_flat_b_res = rgb_imagesres[2].reshape(1, 1);
            Mat matArrayres[] = {m_flat_r_res, m_flat_g_res, m_flat_b_res};
            Mat frameCHWres;
            hconcat(matArrayres, 3, frameCHWres);
            /*convert to FP32*/
            frameCHWres.convertTo(frameCHWres, CV_32FC3);

            /* normailising  pixels */
            divide(frameCHWres, 255.0, frameCHWres);

            /* DRP AI input image should be continuous buffer */
            if (!frameCHWres.isContinuous())
                frameCHWres = frameCHWres.clone();

            Mat frameres = frameCHWres;
            
            auto t1_face = std::chrono::high_resolution_clock::now();

            /* resnet18 inference*/
            runtime1.SetInput(0, frameres.ptr<float>());
            /*inference start time for fairface model*/
            auto t2_face = std::chrono::high_resolution_clock::now();
            runtime1.Run(drpai_freq);
            /*inference end time for fairface model*/
            auto t3_face = std::chrono::high_resolution_clock::now();
            auto inf_duration_face = std::chrono::duration_cast<std::chrono::microseconds>(t3_face - t2_face).count();
        
        /*Postprocess time start for fairface model*/
            auto t4_face = std::chrono::high_resolution_clock::now();
            /*load inference out on drpai_out_buffer*/
            int32_t l = 0;
            int32_t output_num_res = 0;
            std::tuple<InOutDataType, void *, int64_t> output_buffer_res;
            int64_t output_size_res;
            uint32_t size_count_res = 0;

            /* Get the number of output of the target model. */
            output_num_res = runtime1.GetNumOutput();
            size_count_res = 0;
            /*GetOutput loop*/
            for (l = 0; l < output_num_res; l++)
            {
                /* output_buffer below is tuple, which is { data type, address of output data, number of elements } */
                output_buffer_res = runtime1.GetOutput(l);
                /*Output Data Size = std::get<2>(output_buffer). */
                output_size_res = std::get<2>(output_buffer_res);

                /*Output Data Type = std::get<0>(output_buffer)*/
                if (InOutDataType::FLOAT16 == std::get<0>(output_buffer_res))
                {
                    /*Output Data = std::get<1>(output_buffer)*/
                    uint16_t *data_ptr_res = reinterpret_cast<uint16_t *>(std::get<1>(output_buffer_res));
                    for (int j = 0; j < output_size_res; j++)
                    {
                        /*FP16 to FP32 conversion*/
                        drpai_output_buf1[j + size_count_res] = float16_to_float32(data_ptr_res[j]);
                    }
                }
                else if (InOutDataType::FLOAT32 == std::get<0>(output_buffer_res))
                {
                    /*Output Data = std::get<1>(output_buffer)*/
                    float *data_ptr_res = reinterpret_cast<float *>(std::get<1>(output_buffer_res));
                    for (int j = 0; j < output_size_res; j++)
                    {
                        drpai_output_buf1[j + size_count_res] = data_ptr_res[j];
                    }
                }
                else
                {
                    std::cerr << "[ERROR] Output data type : not floating point." << std::endl;
                    ret = -1;
                    break;
                }
                size_count_res += output_size_res;
            }

            if (ret != 0)
            {
                std::cerr << "[ERROR] DRP Inference Not working !!! " << std::endl;
                return -1;
            }
            /*Post process start time for fairface model*/
            R_Post_Proc_ResNet34(drpai_output_buf1, i,instance);
            
            
            /*Postprocess time end for fairface model*/
            auto t5_face = std::chrono::high_resolution_clock::now();

            auto r_post_proc_time_face = std::chrono::duration_cast<std::chrono::microseconds>(t5_face - t4_face).count();
            auto PRE_PROC_TIME_FACE = std::chrono::duration_cast<std::chrono::microseconds>(t1_face - t0_face).count();
            
            POST_PROC_TIME_FACE_MICRO = POST_PROC_TIME_FACE_MICRO + r_post_proc_time_face;
            PRE_PROC_TIME_FACE_MICRO = PRE_PROC_TIME_FACE_MICRO + PRE_PROC_TIME_FACE;
            INF_TIME_FACE_MICRO = INF_TIME_FACE_MICRO + inf_duration_face;

        }
    }
    std::scoped_lock statMutex(inferenceStatistics.mtx);
    auto t6 =  std::chrono::high_resolution_clock::now();
    Push_Statistic(std::chrono::duration_cast<std::chrono::microseconds>(t2-t1), inferenceStatistics.Yolo_preInferenceTime);
    Push_Statistic(std::chrono::duration_cast<std::chrono::microseconds>(t3-t2), inferenceStatistics.Yolo_inferenceTime);
    Push_Statistic(std::chrono::duration_cast<std::chrono::microseconds>(t5-t4), inferenceStatistics.Yolo_postInferenceTime);
    std::chrono::microseconds infTimeFairFace;
    // Make sure we don't plot 0 seconds for inference time
    if(instance.headCount ==  0)
    {
        infTimeFairFace = instance.infTimeTinyFace;
    }
    else
    {
        instance.infTimeTinyFace = std::chrono::duration_cast<std::chrono::microseconds>(t6-t5);
    }

    Push_Statistic(infTimeFairFace, inferenceStatistics.FairFace_inferenceTime);
    /*Calculating the fps*/
    return 0;
}
//#define USE_GSTREAMER
void Face_Detection_Thread(Inference_instance &instance, bool &done)
{
    std::cout << "Started face detect thread " << std::endl;
    while(!done)
    {
        std::unique_lock lock(instance.faceDetectMutex);
        instance.faceDetectWakeUp.wait(lock,[&instance]{return instance.faceDetectQ.size() > 0;  });
        {
      
            if(instance.faceDetectQ.size() > 0 && !instance.faceDetectQ.front().empty() )
            {
                std::scoped_lock lk(drpAIMutex);
                instance.g_frame =  instance.faceDetectQ.front();
                    
                if(!instance.g_frame.empty())
                {
                    try
                    {
                    //std::cout << "Running face detect " << std::endl;
                    Face_Detection(instance.g_frame, instance);
                    }
                    catch(...)
                    {
                        std::cout << "Face detection crash " << std::endl;
                    }
                }
                else
                {
                    std::cout << "Empty frame buffer detected " << std::endl;
                }
                instance.faceDetectQ.pop_front();
            }
        }
    }
}
void Frame_Process_Thread(Inference_instance &instance, bool &done, bool seperateThread)
{
    stringstream stream;
    int32_t inf_sem_check = 0;
    string str = "";
    int runCounter = 0;
    if(1)
    {
        {
            runCounter++;
            

           
            if(!instance.openGLfb || instance.openGLfb->fb.empty())
            {
                return;
            }
            
            //printf("Buffer is at %p with dimensions %d,%d\n",zerocopyfb->fb.ptr(),zerocopyfb->fb.cols, zerocopyfb->fb.rows);
            if (!instance.openGLfb->fb.empty() && input_source == INPUT_SOURCE_MIPI)
            {
                auto t1_ = std::chrono::system_clock::now();
                //cv::imshow("Test", zerocopyfb->fb);
                //cv::waitKey(1000);
                #if USE_DRP_OPENCV_ACCELERATOR
                std::scoped_lock lk(drpAIMutex);
                #endif
                auto t2_ = std::chrono::system_clock::now();
                auto time_diff = t2_ - t1_;
                //std::cout << "Colour conversion time: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_diff).count() << std::endl;
            }
            else
            {
                std::cout << "Empty frame " << std::endl;
                return;
            }
        }

        Size size(DISP_INF_WIDTH, DISP_INF_HEIGHT);
        {
            
            if (instanceIndex == instance.index)
            {
                std::scoped_lock pushLk(instance.faceDetectMutex); // Make sure face detect results aren't modified whilst drawing rectangles
                auto start = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                auto clonedCV = instance.openGLfb->fb;
                instance.faceDetectQ.push_back(clonedCV);
                instance.faceDetectWakeUp.notify_one();
                auto end = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                instanceIndex++;
            }
            else if(instanceIndex > 1)
            {
              instanceIndex++;  
            }
            // // Do not run DRP on every frame, run for 1 frame on the left, 1 frame on the right and then have a break
            if(instanceIndex >= FRAME_SKIPPER) 
                instanceIndex = 0;
            // std::cout << "Inference times for " << instance.name << " is: " << end-start << std::endl;
        }
        auto endCap = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        //std::scoped_lock resultsMutex(instance.faceDetectResultsMutex); // Make sure its safe to read the results
        uint64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        uint64_t td = now_ms - instance.headTimestamp;
        auto clr = Scalar(255, 255, 0); // Red
        stream.str("");
        stream << "#" << instance.index;
        str = stream.str();


        putText(instance.openGLfb->fb, str,Point(50, 50), FONT_HERSHEY_SIMPLEX, 
                                    AGE_CHAR_THICKNESS, clr, 3);
        if (instance.lastHeadCount && td < 2000)
        {
            for (int i = 0; i < instance.headCount; i++)
            {
                cv::Point pt1(instance.cropx1[i], instance.cropy1[i]);
                // and its bottom right corner.
                cv::Point pt2(instance.cropx2[i], instance.cropy2[i]);
                // These two calls...
                // Store a list of rectangles so we can avoid plotting overlapping rectangles
                std::vector<cv::Rect> rectList = std::vector<cv::Rect>();
                rectList.clear();
                {
                    constexpr uint32_t safetyMargin_pixels = 60;
                    uint32_t X_MIN_LIMIT = safetyMargin_pixels;
                    uint32_t Y_MIN_LIMIT = safetyMargin_pixels;
                    uint32_t X_MAX_LIMIT = (instance.openGLfb->fb.cols - safetyMargin_pixels);
                    uint32_t Y_MAX_LIMIT = (instance.openGLfb->fb.rows - safetyMargin_pixels);

                    cv::Rect testRectangle(instance.cropx1[i], instance.cropy1[i], instance.cropx2[i], instance.cropy2[i]);
                    bool overLappingRectangle = false;
                    for (cv::Rect &x : rectList)
                    {
                        // Test if there's an overlapping rectangle already in the list
                        bool overlap = ((testRectangle & x).area() > 0);
                        if (overlap)
                        {
                            std::cout << "Overlap found for rectangle " << std::endl;
                            overLappingRectangle = true;
                        }
                    }
                    rectList.push_back(testRectangle);
                    std::stringstream stream;
                    stream.str("");
                    stream << "Gender: " << instance.gender[i] << std::setw(3);
                    str = stream.str();
                    int x = instance.cropx1[i];
                    int y_gender = instance.cropy1[i];
                    y_gender = y_gender - 80;

                    // Place age/gender above the bounding box
                    // Limit coordinates to prevent crashes
                    if (x < X_MIN_LIMIT)
                    {
                        x = X_MIN_LIMIT;
                    }
                    if (x > X_MAX_LIMIT)
                    {
                        x = X_MAX_LIMIT;
                    }
                    if (y_gender < Y_MIN_LIMIT)
                    {
                        y_gender = Y_MIN_LIMIT;
                        std::cout << "Limiting Y position to be " << y_gender << " Due to limit at " << Y_MIN_LIMIT;
                    }
                    else if (y_gender > Y_MAX_LIMIT)
                    {
                        y_gender = Y_MAX_LIMIT;
                    }

                    int y_age = y_gender + 50;
                    // We can modify the buffer at this point as its not needed anymore
                    putText(instance.openGLfb->fb, str, Point(x, y_gender), FONT_HERSHEY_SIMPLEX,
                            AGE_CHAR_THICKNESS, clr, 3);
                    stream.str("");
                    stream << "Age Group: " << instance.age[i] << std::setw(3);
                    str = stream.str();

                    putText(instance.openGLfb->fb, str, Point(x, y_age), FONT_HERSHEY_SIMPLEX,
                            AGE_CHAR_THICKNESS, clr, 3);
                    cv::rectangle(instance.openGLfb->fb, pt1, pt2, clr, 1.5);
                }
                
            }
        }
        instance.pendingFrameCount++;
        instance.frameCounter++; // Total number of frames
    }
}

void instance_capture_frame(Inference_instance &instance, bool &done)
{
    stringstream stream;
    int32_t inf_sem_check = 0;
    string str = "";
    int32_t ret = 0;
    int32_t baseline = 10;

    cv::Mat g_frame_original;
    cv::Mat g_frame_bgr;

    int wait_key;
    /* Capture stream of frames from camera using Gstreamer pipeline */
    int width = IMAGE_WIDTH;
    int height = IMAGE_HEIGHT;
    
    //instance.frameProcessThread = std::thread(Frame_Process_Thread,std::ref(instance),std::ref(done),true);
    instance.faceDetectThread = std::thread(Face_Detection_Thread,std::ref(instance),std::ref(done));
    #ifndef USE_GSTREAMER
        // Use 15 buffers
        instance.v4lUtil = std::make_shared<V4LUtil>(instance.device,width,height,6, V4L2_PIX_FMT_BGR24);
        std::cout << "Starting Streaming thread for " << instance.name<<  " And pipeline " << gstreamer_pipeline << std::endl;
        instance.v4lUtil->Start();
    #else
        /* Capture stream of frames from camera using Gstreamer pipeline */
        instance.cap.open(instance.gstreamer_pipeline, CAP_GSTREAMER);
    #endif
    while (!done)
    {
        #ifndef USE_GSTREAMER
        auto fb = cv::Mat();
        v4l2_buffer v4lBuffer;
        auto zerocopyFB = instance.v4lUtil->ReadFrame();
        //std::scoped_lock lk(instance.openGLfbMutex);
        if(zerocopyFB == NULL )
        {
            continue;
        }

        #endif

        #ifdef USE_GSTREAMER
        cv::Mat g_frame_original;
        instance.cap >> g_frame_original ;
        #else
        Mat g_frame_original = fb;
        #endif
        auto startCap =  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        instance.openGLfb = zerocopyFB;
        
        auto currrentTime = std::chrono::system_clock::now();
        uint64_t time_difference_microseconds = std::chrono::duration_cast<std::chrono::microseconds> (currrentTime - instance.previousTimestamp).count();
        //std::cout << "Frame period is: " << time_difference_microseconds/1000 << std::endl;
        instance.previousTimestamp = currrentTime;
        instance.frameCounter++; // Total number of frames
        
        try
        {   
            Frame_Process_Thread(instance,done,false);
        }
        catch (...)
        {
            std::cout << "Exception caught " << std::endl;
        }

        instance.Frame_Timestamp.push_back(std::chrono::system_clock::now());
        if (instance.Frame_Timestamp.size() > MAX_STATISTIC_SIZE)
                instance.Frame_Timestamp.pop_front();
    #if 0
        {
            std::scoped_lock instanceThreadLock(instance.frameProcessMutex);
            std::shared_ptr<V4L_ZeroCopyFB> clonedfb = zerocopyFB;
            std::cout << "Reference counter is: " << clonedfb.use_count() << std::endl;
            instance.frameProcessQ.push_back(clonedfb);
            instance.frameProcessThreadWakeup.notify_one();
            std::cout << "Reference counter after is: " << clonedfb.use_count() << std::endl;
        }
        #endif
    }

            
}

#if 0 
/*****************************************
 * Function Name : capture_frame
 * Description   : function to open camera gstreamer pipeline.
 * Arguments     : string cap_pipeline input pipeline
 ******************************************/
void capture_frame(std::string gstreamer_pipeline )
{
    stringstream stream;
    int32_t inf_sem_check = 0;
    string str = "";
    int32_t ret = 0;
    int32_t baseline = 10;
    uint8_t * img_buffer0;
    img_buffer0 = (unsigned char*) (malloc(DISP_OUTPUT_WIDTH*DISP_OUTPUT_HEIGHT*BGRA_CHANNEL));

    cv::Mat g_frame_original;
    cv::Mat g_frame_bgr;

    int wait_key;
    /* Capture stream of frames from camera using Gstreamer pipeline */
    cap.open(gstreamer_pipeline, CAP_GSTREAMER);
    if (!cap.isOpened())
    {
        std::cerr << "[ERROR] Error opening video stream or camera !" << std::endl;
        //return;
        goto ai_inf_end;
    }

    while (true)
    {
        if(input_source == INPUT_SOURCE_USB)
        {
            cap >> g_frame;
        }
        else
        {
            //If input is MIPI need to convert format to BGR
            cap >> g_frame_original;
            cv::cvtColor(g_frame_original, g_frame, cv::COLOR_YUV2BGR_YUY2);
        }

        cv::Mat output_image(DISP_OUTPUT_HEIGHT,DISP_OUTPUT_WIDTH , CV_8UC3, cv::Scalar(0, 0, 0));
        fps = cap.get(CAP_PROP_FPS);
        ret = sem_getvalue(&terminate_req_sem, &inf_sem_check);
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to get Semaphore Value: errno=%d\n", errno);
            goto err;
        }
        
        /*Checks the semaphore value*/
        if (1 != inf_sem_check)
        {
            goto ai_inf_end;
        }
        if (g_frame.empty())
        {
            std::cout << "[INFO] Video ended or corrupted frame !\n";
            continue; //return;
        }
        else
        {
            int ret = Face_Detection();
            if (ret != 0)
            {
                std::cerr << "[ERROR] Inference Not working !!! " << std::endl;
            }
            auto clr = Scalar(0, 255, 0);
            auto now = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count(); 
            if(now - timestamp_detection > AGE_GENDER_DISPLAY_TIMEOUT_MICROSECONDS)
            {
                // Clear the gender/age after 2 seconds of no detection
                gender ="";
                age="";
                clr = Scalar(0,0,255);
            }

            /*Display frame */
            if(1)
            {
            stream.str("");
            stream << "Gender: " << gender << std::setw(3);
            str = stream.str();
            Size tot_time_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, HC_CHAR_THICKNESS, &baseline);
            
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - tot_time_size.width - RIGHT_ALIGN_OFFSET - 50), (AGE_STR_Y + tot_time_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - tot_time_size.width - RIGHT_ALIGN_OFFSET - 50), (AGE_STR_Y + tot_time_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, clr, HC_CHAR_THICKNESS);


            stream.str("");
            stream << "Age Group: "<< age << std::setw(3);
            tot_time_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, HC_CHAR_THICKNESS, &baseline);
            str = stream.str();
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - tot_time_size.width - RIGHT_ALIGN_OFFSET - 50), (AGE_STR_Y + tot_time_size.height+40)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - tot_time_size.width - RIGHT_ALIGN_OFFSET - 50), (AGE_STR_Y + tot_time_size.height+40)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, clr, HC_CHAR_THICKNESS);

            }            

            // stream.str("");
            // stream << "Camera Frame Rate : "<< fixed << setprecision(1) << fps <<" fps ";
            // str = stream.str();
            // Size camera_rate_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, HC_CHAR_THICKNESS, &baseline);
            // putText(output_image, str,Point((DISP_OUTPUT_WIDTH - camera_rate_size.width - RIGHT_ALIGN_OFFSET), (FPS_STR_Y + camera_rate_size.height)), FONT_HERSHEY_SIMPLEX, 
            //             CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
            // putText(output_image, str,Point((DISP_OUTPUT_WIDTH - camera_rate_size.width - RIGHT_ALIGN_OFFSET), (FPS_STR_Y + camera_rate_size.height)), FONT_HERSHEY_SIMPLEX, 
            //             CHAR_SCALE_SMALL, Scalar(255, 255, 255), HC_CHAR_THICKNESS);

            stream.str("");
            stream << "Total Time: " << fixed << setprecision(2)<< TOTAL_TIME <<" ms";
            str = stream.str();
            Size tot_time_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_LARGE, HC_CHAR_THICKNESS, &baseline);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - tot_time_size.width - RIGHT_ALIGN_OFFSET + 10), (T_TIME_STR_Y + tot_time_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_LARGE, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - tot_time_size.width - RIGHT_ALIGN_OFFSET+ 10), (T_TIME_STR_Y + tot_time_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_LARGE, Scalar(0, 255, 0), HC_CHAR_THICKNESS);

            stream.str("");
            stream << "TinyYolov2+FairFace";
            str = stream.str();
            Size tinyyolov2_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, HC_CHAR_THICKNESS, &baseline);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - tinyyolov2_size.width - RIGHT_ALIGN_OFFSET), (MODEL_NAME_1_Y + tinyyolov2_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - tinyyolov2_size.width - RIGHT_ALIGN_OFFSET), (MODEL_NAME_1_Y + tinyyolov2_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(255, 255, 255), HC_CHAR_THICKNESS);

            stream.str("");
            stream << "Pre-Proc: "  << fixed << setprecision(2)<< PRE_PROC_TIME_TINYYOLO+PRE_PROC_TIME_FACE<<" ms";
            str = stream.str();
            Size pre_proc_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, HC_CHAR_THICKNESS, &baseline);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - pre_proc_size.width - RIGHT_ALIGN_OFFSET), (PRE_TIME_STR_Y + pre_proc_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - pre_proc_size.width - RIGHT_ALIGN_OFFSET), (PRE_TIME_STR_Y + pre_proc_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(255, 255, 255), HC_CHAR_THICKNESS);
            stream.str("");
            stream << "Inference: " << fixed << setprecision(2)<< INF_TIME_TINYYOLO+INF_TIME_FACE<<" ms";
            str = stream.str();
            Size inf_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, HC_CHAR_THICKNESS, &baseline);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - inf_size.width - RIGHT_ALIGN_OFFSET), (I_TIME_STR_Y + inf_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - inf_size.width - RIGHT_ALIGN_OFFSET), (I_TIME_STR_Y + inf_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(255, 255, 255), HC_CHAR_THICKNESS);
            stream.str("");
            stream << "Post-Proc: " << fixed << setprecision(2) << POST_PROC_TIME_TINYYOLO+POST_PROC_TIME_FACE <<" ms";
            str = stream.str();
            Size post_proc_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, HC_CHAR_THICKNESS, &baseline);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - post_proc_size.width - RIGHT_ALIGN_OFFSET), (P_TIME_STR_Y + post_proc_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - post_proc_size.width - RIGHT_ALIGN_OFFSET), (P_TIME_STR_Y + post_proc_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(255, 255, 255), HC_CHAR_THICKNESS);
            


            Size size(DISP_INF_WIDTH, DISP_INF_HEIGHT);
            /*resize the image to the keep ratio size*/
            resize(g_frame, g_frame, size);       

            g_frame.copyTo(output_image(Rect(0, 0, DISP_INF_WIDTH, DISP_INF_HEIGHT)));
            cv::Mat bgra_image;
            cv::cvtColor(output_image, bgra_image, cv::COLOR_BGR2BGRA);
            float x1_scaled = (float) ((float)cropx1[0]/(float)IMAGE_WIDTH) * (float)DISP_INF_WIDTH;
            float x2_scaled = (float) ((float)cropx2[0]/(float)IMAGE_WIDTH) * (float)DISP_INF_WIDTH;
            float y1_scaled = (float) ((float)cropy1[0]/(float)IMAGE_HEIGHT) * (float)DISP_INF_HEIGHT;
            float y2_scaled = (float) ((float)cropy2[0]/(float)IMAGE_HEIGHT) * (float)DISP_INF_HEIGHT;
            // Cap coordinates so we don't exceed the boundary
            if(x1_scaled > DISP_INF_WIDTH)
                x1_scaled = DISP_INF_WIDTH;
            if(x2_scaled > DISP_INF_WIDTH)
                x2_scaled = DISP_INF_WIDTH;
            if(y1_scaled > DISP_INF_HEIGHT)
                y1_scaled = DISP_INF_HEIGHT;
            if(y2_scaled > DISP_INF_HEIGHT)
                y2_scaled = DISP_INF_HEIGHT;

            cv::Point start = cv::Point((int)x1_scaled,(int)y1_scaled);
            cv::Point end = cv::Point((int) x2_scaled,(int) y2_scaled);
            cv::Scalar color(255, 0, 0,1.0);  // Blue color
            //std::cout << "Coordinates are X1 : " << x1_scaled << std::endl;
            if(HEAD_COUNT)
                cv::rectangle(bgra_image,start,end,color,2);
            memcpy(img_buffer0, bgra_image.data, DISP_OUTPUT_WIDTH * DISP_OUTPUT_HEIGHT * BGRA_CHANNEL);
            wayland.commit(img_buffer0, NULL);

        }
    }
    free(img_buffer0);
    cap.release(); 
    destroyAllWindows();
    err:
    free(img_buffer0);

    /*Set Termination Request Semaphore to 0*/
    sem_trywait(&terminate_req_sem);
    goto ai_inf_end;
    /*AI Thread Termination*/
    ai_inf_end:
        /*To terminate the loop in Capture Thread.*/
        printf("AI Inference Thread Terminated\n");
        free(img_buffer0);
        pthread_exit(NULL);
        return;
}
#endif

/*****************************************
* Function Name : get_drpai_start_addr
* Description   : Function to get the start address of DRPAImem.
* Arguments     : drpai_fd: DRP-AI file descriptor
* Return value  : If non-zero, DRP-AI memory start address.
*                 0 is failure.
******************************************/

uint64_t get_drpai_start_addr(int drpai_fd)
{
    int ret = 0;
    drpai_data_t drpai_data;

    errno = 0;

    /* Get DRP-AI Memory Area Address via DRP-AI Driver */
    ret = ioctl(drpai_fd , DRPAI_GET_DRPAI_AREA, &drpai_data);
    if (-1 == ret)
    {
        std::cerr << "[ERROR] Failed to get DRP-AI Memory Area : errno=" << errno << std::endl;
        return 0;
    }

    return drpai_data.address;
}




/*****************************************
* Function Name : init_drpai
* Description   : Function to initialize DRP-AI.
* Arguments     : drpai_fd: DRP-AI file descriptor
* Return value  : If non-zero, DRP-AI memory start address.
*                 0 is failure.
******************************************/
uint64_t init_drpai(int drpai_fd)

{
    int ret = 0;

    uint64_t drpai_addr = 0;

    /*Get DRP-AI memory start address*/
    drpai_addr = get_drpai_start_addr(drpai_fd);
    if (drpai_addr == 0)
    {
        return 0;
    }

    return drpai_addr;
}



void *R_Kbhit_Thread(void *threadid)
{
    /*Semaphore Variable*/
    int32_t kh_sem_check = 0;
    /*Variable to store the getchar() value*/
    int32_t c = 0;
    /*Variable for checking return value*/
    int8_t ret = 0;

    printf("Key Hit Thread Starting\n");

    printf("************************************************\n");
    printf("* Press ENTER key to quit. *\n");
    printf("************************************************\n");

    /*Set Standard Input to Non Blocking*/
    errno = 0;
    ret = fcntl(0, F_SETFL, O_NONBLOCK);
    if (-1 == ret)
    {
        fprintf(stderr, "[ERROR] Failed to run fctnl(): errno=%d\n", errno);
        goto err;
    }

    while(1)
    {
        /*Gets the Termination request semaphore value. If different then 1 Termination was requested*/
        /*Checks if sem_getvalue is executed wihtout issue*/
        errno = 0;
        ret = sem_getvalue(&terminate_req_sem, &kh_sem_check);
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to get Semaphore Value: errno=%d\n", errno);
            goto err;
        }
        /*Checks the semaphore value*/
        if (1 != kh_sem_check)
        {
            goto key_hit_end;
        }

        c = getchar();
        if (EOF != c)
        {
            /* When key is pressed. */
            printf("key Detected.\n");
            goto err;
        }
        else
        {
            /* When nothing is pressed. */
            usleep(WAIT_TIME);
        }
    }

/*Error Processing*/
err:
    /*Set Termination Request Semaphore to 0*/
    sem_trywait(&terminate_req_sem);
    goto key_hit_end;

key_hit_end:
    printf("Key Hit Thread Terminated\n");
    pthread_exit(NULL);
}



/*****************************************
 * Function Name : query_device_status
 * Description   : function to check USB device is connected.
 * Arguments     : device_type: for USB,  specify "usb".
 *                         
 * Return value  : media_port, media port that device is connected. 
 ******************************************/
std::string query_device_status(std::string device_type)
{
    std::string media_port = "";
    /* Linux command to be executed */
    const char* command = "v4l2-ctl --list-devices";
    /* Open a pipe to the command and execute it */ 
    FILE* pipe = popen(command, "r");
    if (!pipe) 
    {
        std::cerr << "[ERROR] Unable to open the pipe." << std::endl;
        return media_port;
    }
    /* Read the command output line by line */
    char buffer[128];
    size_t found;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) 
    {
        std::string response = std::string(buffer);
        found = response.find(device_type);
        if (found != std::string::npos)
        {
            fgets(buffer, sizeof(buffer), pipe);
            media_port = std::string(buffer);
            pclose(pipe);
            /* return media port*/
            return media_port;
        } 
    }
    pclose(pipe);
    /* return media port*/
    return media_port;
}

void *R_Inf_Thread(void *threadid)
{
    int8_t ret = 0;
    // ret = wayland.init(DISP_OUTPUT_WIDTH, DISP_OUTPUT_HEIGHT, BGRA_CHANNEL);
    if(0 != ret)
    {
        fprintf(stderr, "[ERROR] Failed to initialize Image for Wayland\n");
       // goto err;
    }
    std::cout << "Done wayland init \n" << std::endl;
    for(int i =0; i< NUM_FRAME_BUFFERS; i ++)
    {
        output_fb[i] = std::vector<uint8_t>(DISP_OUTPUT_WIDTH*DISP_OUTPUT_HEIGHT*BGRA_CHANNEL);
    }
    cv::Mat  output_image(DISP_OUTPUT_HEIGHT,DISP_OUTPUT_WIDTH , CV_8UC3, cv::Scalar(0, 0, 0));
    while(1)
    {
        // Wayland thread, wait for a frame buffer and plot it
        std::unique_lock<std::mutex> lock(output_mutex);
        std::this_thread::sleep_for(1000ms);
        //output_cv.wait(lock, [] { return output_fb_ready[0] && output_fb_ready[1];});

        {
            #if 0 
            std::scoped_lock<mutex> lk(instances[0].scaledFrameMutex);
            std::scoped_lock<mutex> lk2(instances[1].scaledFrameMutex);
            std::cout << "Output FB ready " << output_fb_ready[0]  << std::endl;

            instances[0].scaledFrame.copyTo(output_image(Rect(instances[0].DisplayStartX, instances[0].DisplayStartY,DISP_INF_WIDTH, DISP_INF_HEIGHT)));
            instances[1].scaledFrame.copyTo(output_image(Rect(instances[1].DisplayStartX, instances[1].DisplayStartY, DISP_INF_WIDTH, DISP_INF_HEIGHT)));
            cv::Mat bgra_image;
            cv::cvtColor(output_image, bgra_image, cv::COLOR_BGR2BGRA);
            // Wakeup the wayland thread
            memcpy(output_fb[output_fb_index].data(), bgra_image.data, DISP_OUTPUT_WIDTH * DISP_OUTPUT_HEIGHT * BGRA_CHANNEL);
            wayland.commit(output_fb[output_fb_index].data(), NULL);
            output_fb_index++;
            if(output_fb_index >= NUM_FRAME_BUFFERS)
                output_fb_index = 0;
            output_fb_ready[0] = 0;
            output_fb_ready[1] = 0;
            #endif
        }
    }
/*Error Processing*/
err:
    /*Set Termination Request Semaphore to 0*/
    sem_trywait(&terminate_req_sem);
    goto ai_inf_end;
/*AI Thread Termination*/
ai_inf_end:
    /*To terminate the loop in Capture Thread.*/
    printf("AI Inference Thread Terminated\n");
    pthread_exit(NULL);
}
void Inf_Instance_Capture_Thread(Inference_instance &instance, bool &done)
{
    instance_capture_frame(instance,done);
    /*Error Processing*/
err:
    /*Set Termination Request Semaphore to 0*/
    sem_trywait(&terminate_req_sem);
    goto ai_inf_end;
/*AI Thread Termination*/
ai_inf_end:
    /*To terminate the loop in Capture Thread.*/
    printf("AI Inference Thread Terminated\n");
    pthread_exit(NULL);
}

int8_t R_Main_Process(bool &done, SDL_Window * window,ImVec4& clear_color, bool &show_demo_window, bool &show_another_window, ImGuiIO& io, SDL_GLContext &gl_context)
{
    /*Main Process Variables*/
    int8_t main_ret = 0;
    /*Semaphore Related*/
    int32_t sem_check = 0;
    /*Variable for checking return value*/
    int8_t ret = 0;

    printf("Main Loop Starts\n");
    while(!done)
    {
          // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT)
                done = true;
            if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE && event.window.windowID == SDL_GetWindowID(window))
                done = true;
        }
        if (SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED)
        {
            SDL_Delay(10);
            continue;
        }
        
        {
            std::scoped_lock lk(instances[0].openGLfbMutex);
            std::scoped_lock lk2(instances[1].openGLfbMutex);
            // Start the Dear ImGui frame
            auto start = std::chrono::system_clock::now();
            auto end = std::chrono::system_clock::now();
            start = std::chrono::system_clock::now();
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplSDL2_NewFrame();

            ImGui::NewFrame();
 
            {
     
                LoadTextureFromRGBStream(instances[0]);
                LoadTextureFromRGBStream(instances[1]);
                end = std::chrono::system_clock::now();
                std::string mainName = "Main camera";
                std::string secondName = "Second camera";
                //SHow the stream with the most amount of heads
                if(instances[0].headCount >=  instances[1].headCount)
                {
                    Plot_And_Record_Stream_With_Custom_Shader(instances[0],instances[0].texture,false,mainName);
                    Plot_And_Record_Stream_With_Custom_Shader(instances[1],instances[1].texture,false, secondName);
                }
                else
                {
                    Plot_And_Record_Stream_With_Custom_Shader(instances[1],instances[1].texture,false, mainName);
                    Plot_And_Record_Stream_With_Custom_Shader(instances[0],instances[0].texture,false, secondName);
                }
                
                PlotStatistics(inferenceStatistics);
                PlotFPS(instances[0],instances[1]);
                
                // Rendering        
                ImGui::Render();
                
            }
         
            glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
            glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
            glClear(GL_COLOR_BUFFER_BIT);

            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            SDL_GL_SwapWindow(window);
            
            auto td = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
            std::cout << "Rendering time: " << td << std::endl;
        }
        /*Gets the Termination request semaphore value. If different then 1 Termination was requested*/
        errno = 0;
        ret = sem_getvalue(&terminate_req_sem, &sem_check);
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to get Semaphore Value: errno=%d\n", errno);
            break;
        }
        /*Checks the semaphore value*/
        if (1 != sem_check)
        {
            break;
        }
    }
    std::cout << "Shutting down OpenGL " << std::endl;
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();
}
void Configure_Instances()
{
    std::string media_port0 = "/dev/video0";
    std::string media_port1 = "/dev/video1";
    std::string gstreamer_pipeline_instance0 = "v4l2src device=" + media_port0 +" ! queue ! video/x-raw, width="+std::to_string(1920)+", height="+std::to_string(1080)+" ,framerate=30/1,format=BGR ! queue ! appsink -v";
    std::string gstreamer_pipeline_instance1 = "v4l2src device=" + media_port1 +" ! queue ! video/x-raw, width="+std::to_string(1920)+", height="+std::to_string(1080)+" ,framerate=30/1,format=BGR ! queue ! appsink -v";
            
    instances[0].gstreamer_pipeline = gstreamer_pipeline_instance0;
    instances[0].device = media_port0;
    instances[0].name = "Instance 0";
    instances[0].DisplayStartX = 0;
    instances[0].DisplayStartY = 0;
    instances[0].index = 0;
    // Instance 1
    instances[1].gstreamer_pipeline = gstreamer_pipeline_instance1;
    instances[1].device = media_port1;
    instances[1].name = "Instance 1";
    instances[1].DisplayStartX = DISP_OUTPUT_WIDTH/2;
    instances[1].DisplayStartY = DISP_OUTPUT_HEIGHT/2;
    instances[1].index = 1;
}
int main(int argc, char *argv[])
{
    int32_t create_thread_ai = -1;
    int32_t create_thread_key = -1;
    int8_t ret_main = 0;
    int32_t ret = 0;
    int8_t main_proc = 0;
    int32_t sem_create = -1;
    std::string input_source_str = "MIPI";//argv[1];
    std::cout << "Starting Age and gender detection Application" << std::endl;

    /*Disable OpenCV Accelerator due to the use of multithreading */
    unsigned long OCA_list[16];
    for (int i=0; i < 16; i++)
    {
        OCA_list[i] = USE_DRP_OPENCV_ACCELERATOR;
    }
        // Setup SDL
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0)
    {
        printf("Error: %s\n", SDL_GetError());
        return -1;
    }
    std::cout << "Starting ImGui " << std::endl;

    // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
    // GL ES 2.0 + GLSL 100
    const char* glsl_version = "#version 100";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#elif defined(__APPLE__)
    // GL 3.2 Core + GLSL 150
    const char* glsl_version = "#version 150";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG); // Always required on Mac
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
#else
    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#endif

    // From 2.0.18: Enable native IME.
#ifdef SDL_HINT_IME_SHOW_UI
    SDL_SetHint(SDL_HINT_IME_SHOW_UI, "1");
#endif

    // Create window with graphics context
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_FULLSCREEN);
    SDL_Window* window = SDL_CreateWindow("Dear ImGui SDL2+OpenGL3 example", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1920, 1080, window_flags);
    if (window == nullptr)
    {
        printf("Error: SDL_CreateWindow(): %s\n", SDL_GetError());
        return -1;
    }

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    SDL_GL_MakeCurrent(window, gl_context);
    SDL_GL_SetSwapInterval(1); // Enable vsync

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsLight();

    // Setup Platform/Renderer backends
    ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Load Fonts
    // - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them.
    // - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
    // - If the file cannot be loaded, the function will return a nullptr. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
    // - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
    // - Use '#define IMGUI_ENABLE_FREETYPE' in your imconfig file to use Freetype for higher quality font rendering.
    // - Read 'docs/FONTS.md' for more instructions and details.
    // - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
    // - Our Emscripten build process allows embedding fonts to be accessible at runtime from the "fonts/" folder. See Makefile.emscripten for details.
    //io.Fonts->AddFontDefault();
    //io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\segoeui.ttf", 18.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
    //ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, nullptr, io.Fonts->GetGlyphRangesJapanese());
    //IM_ASSERT(font != nullptr);

    // Our state
    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    glGenTextures(1, &instances[0].texture); 
    glGenTextures(1, &instances[1].texture);
    auto shader = InitCustomShaderProgram(); 
    InitTestImage();
    std::cout << "Adding custom shader " << std::endl;
    
    // Set to full screen
    ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);

#ifdef __EMSCRIPTEN__
    // For an Emscripten build we are disabling file-system access, so let's not attempt to do a fopen() of the imgui.ini file.
    // You may manually call LoadIniSettingsFromMemory() to load settings from your own storage.
    io.IniFilename = nullptr;
    EMSCRIPTEN_MAINLOOP_BEGIN
#else

#endif
    OCA_Activate( &OCA_list[0] );

    drpai_freq = DRPAI_FREQ;

    errno = 0;
    int drpai_fd = open("/dev/drpai0", O_RDWR);
    if (0 > drpai_fd)
    {
        std::cerr << "[ERROR] Failed to open DRP-AI Driver : errno=" << errno << std::endl;
        return -1;
    }
  
    /*Load Label from label_list file*/
    label_file_map = load_label_file(label_list);

    /*Initialzie DRP-AI (Get DRP-AI memory address and set DRP-AI frequency)*/
    drpaimem_addr_start = init_drpai(drpai_fd);

    if (drpaimem_addr_start == 0)
    {
        close(drpai_fd);
        return -1;
    }

    /*Load model_dir structure and its weight to runtime object */
    runtime_status = runtime.LoadModel(model_dir, drpaimem_addr_start + DRPAI_MEM_OFFSET1);
    
    if(!runtime_status)
    {
        std::cerr << "[ERROR] Failed to load model. " << std::endl;
        close(drpai_fd);
        return -1;
    }    

    std::cout << "[INFO] loaded runtime model :" << model_dir << "\n\n";

     
     /*Load model_dir structure and its weight to runtime object */
    runtime_status1 = runtime1.LoadModel(model_dir1, drpaimem_addr_start + DRPAI_MEM_OFFSET);
    
    if(!runtime_status1)
    {
        std::cerr << "[ERROR] Failed to load model. " << std::endl;
        close(drpai_fd);
        return -1;
    }    

    std::cout << "[INFO] loaded runtime model :" << model_dir1 << "\n\n";

    /* mipi source not supprted */ 
    Configure_Instances(); // Configures Instances 0 and 1
    switch (input_source_map[input_source_str])
    {
        /* Input Source : USB*/
        case INPUT_SOURCE_USB:
        {
            std::cout << "[INFO] USB CAMERA \n";
            input_source = INPUT_SOURCE_USB;
            media_port = query_device_status("usb");
            gstreamer_pipeline = "v4l2src device=" + media_port + " ! video/x-raw, width=640, height=480 ! videoconvert ! appsink";
            sem_create = sem_init(&terminate_req_sem, 0, 1);
            if (0 != sem_create)
            {
                fprintf(stderr, "[ERROR] Failed to Initialize Termination Request Semaphore.\n");
                ret_main = -1;
                goto end_threads;
            }

            create_thread_key = pthread_create(&kbhit_thread, NULL, R_Kbhit_Thread, NULL);
            if (0 != create_thread_key)
            {
                fprintf(stderr, "[ERROR] Failed to create Key Hit Thread.\n");
                ret_main = -1;
                goto end_threads;
            }

            create_thread_ai = pthread_create(&ai_inf_thread, NULL, R_Inf_Thread, NULL);
            if (0 != create_thread_ai)
            {
                sem_trywait(&terminate_req_sem);
                fprintf(stderr, "[ERROR] Failed to create AI Inference Thread.\n");
                ret_main = -1;
                goto end_threads;
            }
        }
        break;

        case INPUT_SOURCE_MIPI:
        {
            std::cout << "[INFO] MIPI CAMERA \n";
            input_source = INPUT_SOURCE_MIPI;
            media_port = query_device_status("RZG2L_CRU");
            gstreamer_pipeline = "v4l2src device=" + media_port +" ! video/x-raw, width="+std::to_string(1920)+", height="+std::to_string(1080)+" ,framerate=30/1 ! videoconvert ! video/x-raw,format=YUY2,width=1920,height=1080,framerate=30/1 ! appsink -v";
            
            sem_create = sem_init(&terminate_req_sem, 0, 1);
            if (0 != sem_create)
            {
                fprintf(stderr, "[ERROR] Failed to Initialize Termination Request Semaphore.\n");
                ret_main = -1;
                goto end_threads;
            }
            std::cout << "Starting inference thread " << std::endl;
            std::thread instance_0 = std::thread(Inf_Instance_Capture_Thread, std::ref(instances[0]),std::ref(thread_done));
            instance_0.detach();
            std::thread instance_1 = std::thread(Inf_Instance_Capture_Thread, std::ref(instances[1]), std::ref(thread_done));
            instance_1.detach();
        }
        break;

        default:
        {
            fprintf(stderr, "[ERROR] Invalid input source mapping %d\n",input_source_map[input_source_str]);
        }
        break;
    }   
    std::cout << "Starting main process" << std::endl;

    main_proc = R_Main_Process(thread_done,window,clear_color,show_demo_window,show_another_window,io, gl_context);
        if (0 != main_proc)
        {
            fprintf(stderr, "[ERROR] Error during Main Process\n");
            ret_main = -1;
        }
        goto end_threads;

 end_threads:

    if (0 == create_thread_ai)
    {
        ret = wait_join(&ai_inf_thread, AI_THREAD_TIMEOUT);
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to exit AI Inference Thread on time.\n");
            ret_main = -1;
        }
    }
    if (0 == create_thread_key)
    {
        ret = wait_join(&kbhit_thread, KEY_THREAD_TIMEOUT);
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to exit Key Hit Thread on time.\n");
            ret_main = -1;
        }
    }

    if (0 == sem_create)
    {
        sem_destroy(&terminate_req_sem);
    }
    /* Exit the program */
    wayland.exit();
    close(drpai_fd);
    return 0;

}
