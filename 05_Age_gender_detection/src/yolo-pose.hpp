#pragma once
#include <opencv2/opencv.hpp>


cv::Mat Run_Yolo_Pose(cv::Mat &input_image, MeraDrpRuntimeWrapper &runtime, int drpai_freq, int INPUT_WIDTH, int INPUT_HEIGHT);