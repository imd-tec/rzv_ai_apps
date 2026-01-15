#include <opencv2/opencv.hpp>
#include "MeraDrpRuntimeWrapper.h"
#include <stdint.h>
#include <chrono>
#include <math.h>
#include <tuple>
#include <builtin_fp16.h>
/*TVM: information*/
#define TVM_MODEL_IN_W              (256)
#define TVM_MODEL_IN_H              (256)
#define TVM_MODEL_OUT_NUM           (1)
#define TVM_MODEL_OUT_W             (256)
#define TVM_MODEL_OUT_H             (256)
#define NUM_CLASS                   (1)
float float16_to_float32(uint16_t a);

int8_t get_result(MeraDrpRuntimeWrapper &runtime, cv::Mat &result)
{
    int8_t ret = 0;
    int32_t output_num = 0;
    std::tuple<InOutDataType, void*, int64_t> output_buffer;
    int64_t output_size;
    cv::Mat depth_float = cv::Mat(TVM_MODEL_OUT_H, TVM_MODEL_OUT_W, CV_32FC1);

    /* Get the number of output of the target model. */
    output_num = runtime.GetNumOutput();
    if(output_num != TVM_MODEL_OUT_NUM){
        return -1;
    }

    /* output_buffer below is tuple, which is { data type, address of output data, number of elements } */
    output_buffer = runtime.GetOutput(0);
    /*Output Data Size = std::get<2>(output_buffer). */
    output_size = std::get<2>(output_buffer);
    if(output_size != TVM_MODEL_OUT_W * TVM_MODEL_OUT_H * NUM_CLASS){
        return -1;
    }

    /*Output Data Type = std::get<0>(output_buffer)*/
    if (InOutDataType::FLOAT16 == std::get<0>(output_buffer))
    {
        /*Output Data = std::get<1>(output_buffer)*/
        
        uint16_t* data_ptr = reinterpret_cast<uint16_t*>(std::get<1>(output_buffer));
        for (int j = 0; j<output_size; j++)
        {
            /*FP16 to FP32 conversion*/
            depth_float.ptr<float>()[j] = float16_to_float32(data_ptr[j]);
        }
    }
    else if (InOutDataType::FLOAT32 == std::get<0>(output_buffer))
    {
        /*Output Data = std::get<1>(output_buffer)*/
        float* data_ptr = reinterpret_cast<float*>(std::get<1>(output_buffer));
        for (int j = 0; j<output_size; j++)
        {
            depth_float.ptr<float>()[j] = data_ptr[j];
        }
    }
    else
    {
        fprintf(stderr, "[ERROR] Output data type : not floating point.\n");
        ret = -1;
    }
    // Convert grayscale float to a displayable format
    cv::normalize(depth_float, result, 0., 255., cv::NORM_MINMAX, CV_8UC1); // Normalize to 0-255

    return ret;
}
cv::Mat Run_MIDAS_Depth_Model(cv::Mat &input_image, MeraDrpRuntimeWrapper &runtime, int drpai_freq, int MIDAS_INPUT_WIDTH, int MIDAS_INPUT_HEIGHT)
{
    // Preprocess input image
    cv::Mat resized_image;
    cv::Mat output_image;
    cv::resize(input_image, resized_image, cv::Size(MIDAS_INPUT_WIDTH, MIDAS_INPUT_HEIGHT));
    cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);
    resized_image.convertTo(resized_image, CV_32FC3, 1.0 / 255);

    // Set input for Midas model
    runtime.SetInput(0, resized_image.ptr<float>());

    // Inference start time for Midas model
    auto t2_midas = std::chrono::high_resolution_clock::now();
    runtime.Run(drpai_freq);
    // Inference end time for Midas model
    auto t3_midas = std::chrono::high_resolution_clock::now();
    auto inf_duration_midas = std::chrono::duration_cast<std::chrono::microseconds>(t3_midas - t2_midas).count();
    std::cout << "[INFO] Midas Inference Time: " << inf_duration_midas << " us" << std::endl;


    // Postprocess time start for Midas model
    auto t4_midas = std::chrono::high_resolution_clock::now();
    // Load inference output on drpai_output_buf2
    cv::Mat depth_map;
    int8_t ret = get_result(runtime, depth_map);
    if (ret != 0)
    {
        std::cerr << "[ERROR] DRP Inference Not working !!! " << std::endl;
        return cv::Mat();
    }
    return depth_map;

}