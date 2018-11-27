//
// Created by zqp on 18-11-25.
//

#ifndef LSTMCHARSRECOGNIZE_CHARRECOGNIZE_H
#define LSTMCHARSRECOGNIZE_CHARRECOGNIZE_H

#include "ICharsRecognize.h"
#include <caffe/caffe.hpp>
using namespace std;

class CharsRecognize: public ICharsRecognize
{
public:
    CharsRecognize(const string& deploy_file,const string& weight_file,const string& mean_file,const vector<float> &mean_values,const string& label_file,const int gpu_id);
    virtual std::pair<std::string,float> recognize(const cv::Mat &im);


private:
    void setMean(const string& mean_file);
    void setMean(const vector<float> &mean_values);
    void wrapInputLayer(const cv::Mat &im);
    cv::Mat imResize(const cv::Mat &im);

    string getPredictString(const vector<float>& fm);

    void getLayerFeatureMaps(string &layer_name,vector<float> &out_data,vector<int>& out_shape);
    cv::Mat imConvert(const cv::Mat &im);

private:
    int gpu_id = 0;
    std::shared_ptr<caffe::Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_ = 1;
    cv::Mat mean_;
    vector<string> labels_;
    int index_blank = -1;
    cv::Scalar channel_mean;
    bool use_mean_file = true;
    int imgtype;
};

#endif //LSTMCHARSRECOGNIZE_CHARRECOGNIZE_H
