//
// Created by zqp on 18-11-25.
//

#ifndef LSTMCHARSRECOGNIZE_ICHARSRECOGNIZE_H
#define LSTMCHARSRECOGNIZE_ICHARSRECOGNIZE_H

#include <string>
#include <opencv2/opencv.hpp>


class ICharsRecognize
{
public:
    virtual std::pair<std::string,float> recognize(const cv::Mat &im)=0;
    ~ICharsRecognize(){}

};

ICharsRecognize *CreateICharsRecognize(const std::string& model_dir,const int gpu_id);

#endif //LSTMCHARSRECOGNIZE_ICHARSRECOGNIZE_H
