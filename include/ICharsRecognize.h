//
// Created by zqp on 18-11-25.
//

#ifndef LSTMCHARSRECOGNIZE_ICHARSRECOGNIZE_H
#define LSTMCHARSRECOGNIZE_ICHARSRECOGNIZE_H

#include <string>
#include <opencv2/opencv.hpp>

using namespace std;


class ICharsRecognize
{
public:
    virtual std::pair<std::string,float> recognize(const cv::Mat &im)=0;
    ~ICharsRecognize(){}

};

ICharsRecognize *CreateICharsRecognize(const std::string& deploy_file,const string weight_file,
                                       const string& label_file,const vector<float>& mean_value, const string mean_file="",const int gpu_id=0);

#endif //LSTMCHARSRECOGNIZE_ICHARSRECOGNIZE_H
