#pragma once
#include <opencv2/opencv.hpp>

/** Runs the midas depth model */
cv::Mat Run_MIDAS_Depth_Model(cv::Mat &input_image, MeraDrpRuntimeWrapper &runtime2, int drpai_freq, int MIDAS_INPUT_WIDTH, int MIDAS_INPUT_HEIGHT);