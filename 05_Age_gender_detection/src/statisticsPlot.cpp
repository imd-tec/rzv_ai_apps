#include "inference.hpp"
#include "imgui.h"
#include "implot.h"
#include <deque>
#include <chrono>

std::vector<int> ConvertListToVectorOfms(std::list<std::chrono::microseconds> &l)
{
    std::list<std::chrono::microseconds>::iterator it;
    std::vector<int> time_ms = std::vector<int>();
    
    for (it = l.begin(); it != l.end(); ++it){
        time_ms.push_back(it->count()/1000);
    }
    return time_ms;

}

void PlotStatistics( Inference_Statistics &stats)
{
    std::scoped_lock statMutex(stats.mtx);
    std::string stats_window_name = "Inference times";
    ImGui::Begin(stats_window_name.c_str());
    if (stats.Yolo_preInferenceTime.size() >= MAX_STATISTIC_SIZE)
    {
        if (ImPlot::BeginPlot("Frame Statistics"))
        {
            ImPlot::SetupAxis(ImAxis_X1, "50 most recent frames");
            ImPlot::SetupAxis(ImAxis_Y1, "Time (ms)");
            uint32_t numPlots = 50;
            ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, 0, 20);
            auto yolo_inf = ConvertListToVectorOfms(stats.Yolo_inferenceTime);
            auto yolo_pre = ConvertListToVectorOfms(stats.Yolo_preInferenceTime);
            auto yolo_post = ConvertListToVectorOfms(stats.Yolo_postInferenceTime);
            auto FairFace_inf = ConvertListToVectorOfms(stats.FairFace_inferenceTime);

            ImPlot::PlotLine("Yolo: Total Inference time", yolo_inf.data(), yolo_inf.size());
            ImPlot::PlotLine("Fair face: Total Inference time", FairFace_inf.data(),FairFace_inf.size());


            int microsecondResults[1] =  { *yolo_inf.end()};

           // ImPlot::PlotBars("Yolo Inference",&yolo_inf[49],1,0.4,1);
            // ImPlot::PlotBars("Fair Face Inference",&FairFace_inf[49],1,0.4,1);
            ImPlot::EndPlot();
        }

    }
    ImGui::End();
}

std::vector<float> Calculate_FPS(Inference_instance &instance)
{
    std::list<std::chrono::system_clock::time_point>::iterator it;
    std::vector<float> time_ms = std::vector<float>();
    std::chrono::system_clock::time_point previousTimePoint = *instance.Frame_Timestamp.begin();
    for (it = std::next(instance.Frame_Timestamp.begin()); it != instance.Frame_Timestamp.end(); ++it){
        uint32_t timeDifference = std::chrono::duration_cast<std::chrono::microseconds> (*it - previousTimePoint).count();
        time_ms.push_back((float)1E6/timeDifference);
        //std::cout << "Time difference: " << timeDifference << std::endl;
        previousTimePoint = *it;
    }
    return time_ms;
}
std::vector<float> reduceArrayByRollingAverage(const std::vector<float>& originalArray) {
    int windowSize = 10;  // The window size for averaging
    int reducedSize = originalArray.size() / windowSize;  // Reduced array size

    // Vector to hold the reduced array
    std::vector<float> reducedArray(reducedSize);

    float totalFPS = 0;

    // Iterate over the original array in steps of 10 (windowSize)
    for (int i = 0; i < reducedSize; ++i) {
        float sum = 0;
        // Calculate the sum of the current window
        for (int j = 0; j < windowSize; ++j) {
            sum += originalArray[i * windowSize + j];
            totalFPS+=originalArray[i * windowSize + j];
        }
        // Store the average of the window in the reduced array
        reducedArray[i] = sum / windowSize;
        //std::cout << "Value is " << reducedArray[i] << std::endl;
    }
   // std::cout << "Average is " << totalFPS/originalArray.size();
    return reducedArray;
}
void PlotFPS( Inference_instance &instance0, Inference_instance &instance1)
{
    std::scoped_lock statMutex(instance0.timestampMtx);
    std::scoped_lock statMutex2(instance1.timestampMtx);
    std::string stats_window_name = "FPS";
    ImGui::Begin(stats_window_name.c_str());
    if (instance0.Frame_Timestamp.size() >= MAX_STATISTIC_SIZE)
    {
        if (ImPlot::BeginPlot("FPS statistics"))
        {
            ImPlot::SetupAxis(ImAxis_X1, "20 most recent frames");
            ImPlot::SetupAxis(ImAxis_Y1, "FPS (Hz)");
            ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, 0, 20);
            uint32_t numPlots = 50;

            auto fps_0 = Calculate_FPS(instance0);
            auto fps_1 = Calculate_FPS(instance1);
            
         

            ImPlot::PlotLine("Instance 0", fps_0.data(), fps_0.size());
            ImPlot::PlotLine("Instance 1", fps_1.data(), fps_1.size());


           // ImPlot::PlotBars("Yolo Inference",&yolo_inf[49],1,0.4,1);
            // ImPlot::PlotBars("Fair Face Inference",&FairFace_inf[49],1,0.4,1);
            ImPlot::EndPlot();
        }

    }
    ImGui::End();
}
