// SamProject.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include <iostream>
#include "sam.h"
#include "SJSegmentAnything.h"
#include "SJSegmentAnythingGPU.h"
#include "opencv2/opencv.hpp"
#include <Windows.h>
using namespace std;
void PerfomanceTest()
{
    LARGE_INTEGER tickFreq;
    LARGE_INTEGER tickStart;
    LARGE_INTEGER tickEnd;
    QueryPerformanceFrequency(&tickFreq);

    double res;
    cv::Size inputSize;
    cv::Mat image;
    std::vector<cv::Point> points;
    cv::Mat mask;

    image = cv::imread("..\\..\\Data\\000.png");
    points.clear();
    points.push_back(cv::Point(568, 305));

    SJSegmentAnything* samcpu;
    SJSegmentAnythingGPU* samgpu;
    //SJSegmentAnythingTRT* samtrt;
    QueryPerformanceCounter(&tickStart);
    samcpu = new SJSegmentAnything();
    samcpu->InitializeSamModel("..\\..\\\models\\sam_onnx_preprocess.onnx", "..\\..\\models\\sam_onnx_example.onnx");
    inputSize = samcpu->GetInputSize();
    cv::resize(image, image, inputSize);
    mask = cv::Mat(inputSize.height, inputSize.width, CV_8UC1);
    samcpu->SamLoadImage(image);
    samcpu->GetMask(points, {}, {}, mask, res);
    delete samcpu;
    QueryPerformanceCounter(&tickEnd);
    cout << "Warming up (CPU) : " << (double)(tickEnd.QuadPart - tickStart.QuadPart) / (double)(tickFreq.QuadPart) << "sec" << endl;

    QueryPerformanceCounter(&tickStart);
    samcpu = new SJSegmentAnything();
    samcpu->InitializeSamModel("..\\..\\\models\\sam_onnx_preprocess.onnx", "..\\..\\models\\sam_onnx_example.onnx");
    inputSize = samcpu->GetInputSize();
    cv::resize(image, image, inputSize);
    mask = cv::Mat(inputSize.height, inputSize.width, CV_8UC1);
    samcpu->SamLoadImage(image);
    samcpu->GetMask(points, {}, {}, mask, res);
    QueryPerformanceCounter(&tickEnd);
    cout << "Second (CPU): " << (double)(tickEnd.QuadPart - tickStart.QuadPart) / (double)(tickFreq.QuadPart) << "sec" << endl;

    QueryPerformanceCounter(&tickStart);
    for (int i = 0; i < 100; i++) {
        samcpu->GetMask(points, {}, {}, mask, res);
    }
    QueryPerformanceCounter(&tickEnd);

    cout << "GetMask Only (CPU) " << (double)(tickEnd.QuadPart - tickStart.QuadPart) / (double)(tickFreq.QuadPart) << "sec" << endl;
    delete samcpu;



    QueryPerformanceCounter(&tickStart);
    samgpu = new SJSegmentAnythingGPU();
    samgpu->InitializeSamModel("..\\..\\\models\\sam_onnx_preprocess.onnx", "..\\..\\models\\sam_onnx_example.onnx");
    inputSize = samgpu->GetInputSize();
    cv::resize(image, image, inputSize);
    mask = cv::Mat(inputSize.height, inputSize.width, CV_8UC1);
    samgpu->SamLoadImage(image);
    samgpu->GetMask(points, {}, {}, mask, res);
    delete samgpu;
    QueryPerformanceCounter(&tickEnd);
    cout << "Warming up (GPU) : " << (double)(tickEnd.QuadPart - tickStart.QuadPart) / (double)(tickFreq.QuadPart) << "sec" << endl;

    QueryPerformanceCounter(&tickStart);
    samgpu = new SJSegmentAnythingGPU();
    samgpu->InitializeSamModel("..\\..\\\models\\sam_onnx_preprocess.onnx", "..\\..\\models\\sam_onnx_example.onnx");
    inputSize = samgpu->GetInputSize();
    cv::resize(image, image, inputSize);
    mask = cv::Mat(inputSize.height, inputSize.width, CV_8UC1);
    samgpu->SamLoadImage(image);
    samgpu->GetMask(points, {}, {}, mask, res);
    QueryPerformanceCounter(&tickEnd);
    cout << "Second (GPU): " << (double)(tickEnd.QuadPart - tickStart.QuadPart) / (double)(tickFreq.QuadPart) << "sec" << endl;

    QueryPerformanceCounter(&tickStart);
    for (int i = 0; i < 100; i++) {
        samgpu->GetMask(points, {}, {}, mask, res);
    }
    QueryPerformanceCounter(&tickEnd);
    cout << "GetMask Only (GPU) " << (double)(tickEnd.QuadPart - tickStart.QuadPart) / (double)(tickFreq.QuadPart) << "sec" << endl;
    delete samgpu;

    QueryPerformanceCounter(&tickStart);
    samcpu = new SJSegmentAnything();
    samcpu->InitializeSamModel("..\\..\\models\\sam_onnx_preprocess.onnx", "..\\..\\models\\sam_onnx_example.onnx");
    inputSize = samcpu->GetInputSize();
    cv::resize(image, image, inputSize);
    mask = cv::Mat(inputSize.height, inputSize.width, CV_8UC1);
    samcpu->SamLoadImage(image);
    samcpu->GetMask(points, {}, {}, mask, res);
    delete samcpu;
    QueryPerformanceCounter(&tickEnd);
    cout << "Warming up (CPU) : " << (double)(tickEnd.QuadPart - tickStart.QuadPart) / (double)(tickFreq.QuadPart) << "sec" << endl;

    QueryPerformanceCounter(&tickStart);
    samcpu = new SJSegmentAnything();
    samcpu->InitializeSamModel("..\\..\\models\\sam_onnx_preprocess.onnx", "..\\..\\models\\sam_onnx_example.onnx");
    inputSize = samcpu->GetInputSize();
    cv::resize(image, image, inputSize);
    mask = cv::Mat(inputSize.height, inputSize.width, CV_8UC1);
    samcpu->SamLoadImage(image);
    samcpu->GetMask(points, {}, {}, mask, res);
    QueryPerformanceCounter(&tickEnd);
    cout << "Second (CPU): " << (double)(tickEnd.QuadPart - tickStart.QuadPart) / (double)(tickFreq.QuadPart) << "sec" << endl;

    QueryPerformanceCounter(&tickStart);
    for (int i = 0; i < 100; i++) {
        samcpu->GetMask(points, {}, {}, mask, res);
    }
    QueryPerformanceCounter(&tickEnd);

    cout << "GetMask Only (CPU) " << (double)(tickEnd.QuadPart - tickStart.QuadPart) / (double)(tickFreq.QuadPart) << "sec" << endl;
    delete samcpu;

    /*QueryPerformanceCounter(&tickStart);
    samtrt = new SJSegmentAnythingTRT();
    samtrt->InitializeSamModel("..\\..\\models\\sam_onnx_preprocess.onnx", "..\\..\\models\\sam_onnx_example.onnx");
    inputSize = samtrt->GetInputSize();
    cv::resize(image, image, inputSize);
    mask = cv::Mat(inputSize.height, inputSize.width, CV_8UC1);
    samtrt->SamLoadImage(image);
    samtrt->GetMask(points, {}, {}, mask, res);
    delete samtrt;
    QueryPerformanceCounter(&tickEnd);
    cout << "Warming up (TRT) : " << (double)(tickEnd.QuadPart - tickStart.QuadPart) / (double)(tickFreq.QuadPart) << "sec" << endl;

    QueryPerformanceCounter(&tickStart);
    samtrt = new SJSegmentAnythingTRT();
    samtrt->InitializeSamModel("..\\..\\models\\sam_onnx_preprocess.onnx", "..\\..\\models\\sam_onnx_example.onnx");
    inputSize = samtrt->GetInputSize();
    cv::resize(image, image, inputSize);
    mask = cv::Mat(inputSize.height, inputSize.width, CV_8UC1);
    samtrt->SamLoadImage(image);
    samtrt->GetMask(points, {}, {}, mask, res); 
    QueryPerformanceCounter(&tickEnd);
    cout << "Second (TRT): " << (double)(tickEnd.QuadPart - tickStart.QuadPart) / (double)(tickFreq.QuadPart) << "sec" << endl;

    QueryPerformanceCounter(&tickStart);
    for (int i = 0; i < 100; i++) {
        samtrt->GetMask(points, {}, {}, mask, res);
    }
    QueryPerformanceCounter(&tickEnd);

    cout << "GetMask Only (TRT) " << (double)(tickEnd.QuadPart - tickStart.QuadPart) / (double)(tickFreq.QuadPart) << "sec" << endl;
    delete samtrt;*/
}
void SamOriginal()
{
    Sam::Parameter param("..\\..\\\models\\sam_onnx_preprocess.onnx", "..\\..\\models\\sam_onnx_example.onnx", std::thread::hardware_concurrency());
    param.providers[0].deviceType = 1; // cpu for preprocess
    param.providers[1].deviceType = 1; // CUDA for sam
    Sam sam(param);
    auto inputSize = sam.getInputSize();
    cv::Mat image = cv::imread("..\\..\\Data\\000.png");
    cv::resize(image, image, inputSize);
    sam.loadImage(image);
    //cv::Mat mask = sam.autoSegment({ 10, 10 });
    cv::Mat mask = sam.getMask({ 568, 305 }); // 533 * 1024 / 960, 286 * 1024 / 960

    cv::imwrite("output_original.png", mask);
}
void SamCPU()
{
    double res;
    cv::Size inputSize;
    cv::Mat image;
    std::vector<cv::Point> points;
    cv::Mat mask;

    image = cv::imread("..\\..\\Data\\000.png");
    points.clear();
    points.push_back(cv::Point(568, 305));

    SJSegmentAnything* samcpu;
    samcpu = new SJSegmentAnything();
    samcpu->InitializeSamModel("..\\..\\models\\sam_onnx_preprocess.onnx", "..\\..\\models\\sam_onnx_example.onnx");
    inputSize = samcpu->GetInputSize();
    cv::resize(image, image, inputSize);
    mask = cv::Mat(inputSize.height, inputSize.width, CV_8UC1);
    samcpu->SamLoadImage(image);
    samcpu->GetMask(points, {}, {}, mask, res);
    cv::imwrite("output_cpu.png", mask);

    delete samcpu;

}
void SamGPU()
{
    double res;
    cv::Size inputSize;
    cv::Mat image;
    std::vector<cv::Point> points;
    cv::Mat mask;

    image = cv::imread("..\\..\\Data\\000.png");
    points.clear();
    points.push_back(cv::Point(568, 305));

    SJSegmentAnythingGPU* samgpu;
    samgpu = new SJSegmentAnythingGPU();
    samgpu->InitializeSamModel("..\\..\\models\\sam_onnx_preprocess.onnx", "..\\..\\models\\sam_onnx_example.onnx");
    inputSize = samgpu->GetInputSize();
    cv::resize(image, image, inputSize);
    mask = cv::Mat(inputSize.height, inputSize.width, CV_8UC1);
    samgpu->SamLoadImage(image);
    samgpu->GetMask(points, {}, {}, mask, res);
    cv::imwrite("output_gpu.png", mask);

    delete samgpu;
}
/*void SamTRT()
{
    double res;
    cv::Size inputSize;
    cv::Mat image;
    std::vector<cv::Point> points;
    cv::Mat mask;

    image = cv::imread("..\\..\\Data\\000.png");
    points.clear();
    points.push_back(cv::Point(568, 305));

    SJSegmentAnythingTRT *samtrt;
    samtrt = new SJSegmentAnythingTRT();
    //printf("%d\n", samtrt->InitializeSamModel("..\\\models\\sam_onnx_preprocess.onnx", "..\\models\\sam_onnx_example.onnx"));
    printf("%d\n", samtrt->InitializeSamModel("..\\..\\models\\sam_vit_h_embedding_first.onnx", "..\\..\\models\\sam_vit_h_embedding_second.onnx"));
    inputSize = samtrt->GetInputSize();
    cv::resize(image, image, inputSize);
    mask = cv::Mat(inputSize.height, inputSize.width, CV_8UC1);
    samtrt->SamLoadImage(image);
    samtrt->GetMask(points, {}, {}, mask, res);
    cv::imwrite("output_trt.png", mask);

    delete samtrt;
}*/
int main()
{   
    //SamOriginal();
    //SamCPU();
    //SamGPU();
    //SamTRT();
    PerfomanceTest();

    /*Sam::Parameter param("..\\\models\\sam_onnx_preprocess.onnx", "..\\models\\sam_onnx_example.onnx", std::thread::hardware_concurrency());
    param.providers[0].deviceType = 1; // cpu for preprocess
    param.providers[1].deviceType = 1; // CUDA for sam
    Sam sam(param);
    printf("here!!\n");
    auto inputSize = sam.getInputSize();
    printf("here!!\n");

    cv::Mat image = cv::imread("..\\Data\\000.png");
    printf("here!!\n");

    cv::resize(image, image, inputSize);
    printf("here!!\n");

    sam.loadImage(image);
    printf("Finish!!\n");

    //cv::Mat mask = sam.autoSegment({ 10, 10 });
    cv::Mat mask = sam.getMask({ 568, 305 }); // 533 * 1024 / 960, 286 * 1024 / 960

    cv::imwrite("output.png", mask);*/
    //SJSegmentAnything sam;
    
}