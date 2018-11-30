//
// Created by zqp on 18-11-25.
//
#include <CharsRecognize.h>
#include <ctcpp.h>

using namespace caffe;
using namespace cv;

float CharsRecognize::getCTCLoss(const float *data, const int timesteps, const vector<int> &predicts)
{
    size_t workspace_alloc_bytes_;
    ctcOptions options;
    options.loc = CTC_CPU;
    options.num_threads = 8;
    options.blank_label = index_blank;

    int len = int(predicts.size());
    ctcStatus_t status = CTC::get_workspace_size<float>(&len, &timesteps,
            int(labels_.size()),1,options,&workspace_alloc_bytes_);

    vector<float> workspace_(workspace_alloc_bytes_);

    float cost = 0;
    status = CTC::compute_ctc_loss_cpu<float>(data,0,predicts.data(),
            &len,&timesteps,labels_.size(),1,&cost,workspace_.data(),options);

    return cost;
}
CharsRecognize::CharsRecognize(const string &deploy_file, const string &weight_file, const string &mean_file,
                               const vector<float> &mean_values, const string &label_file, const int gpu_id)
{
    if(gpu_id<0)
        Caffe::set_mode(Caffe::CPU);
    else
    {
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(gpu_id);
    }

//Load network
    net_.reset(new Net<float>(deploy_file,TEST));
    net_->CopyTrainedLayersFrom(weight_file);

#if 1
    vector<string> blob_names = net_->blob_names();
    for (int i=0; i<blob_names.size(); ++i)
    {
        const boost::shared_ptr<Blob<float> > blob = net_->blob_by_name(blob_names[i]);
        int n = blob->num();
        int c = blob->channels();
        int h = blob->height();
        int w = blob->width();
        cout<<blob_names[i]<<" : "<<n<<" "<<c<<" "<<h<<" "<<w<<endl;
    }
#endif

    CHECK_EQ(net_->num_inputs(),1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(),1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    imgtype = num_channels_==3 ? CV_32FC3 : CV_32FC1;

    CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";

    input_geometry_ = cv::Size(input_layer->width(),input_layer->height());

//  Load mean
    if(mean_file=="")
    {
        setMean(mean_values);
        use_mean_file = false;
    }
    else
    {
        setMean(mean_file);
        use_mean_file = true;
    }

// Load labels
    if(label_file.size())
    {
        std::ifstream labels(label_file.c_str());
        CHECK(labels) << "Unable to open labels file " << label_file;
        string line;
        while (std::getline(labels, line))
        {
            labels_.push_back(string(line));
            if(line=="blank")
                index_blank = labels_.size() - 1;

        }
    }
    else
    {
        Blob<float>* output_layer = net_->output_blobs()[0];
        char szlabel[100];
        printf("output ch=%d\n", output_layer->channels());
        for (int i = 0; i < output_layer->channels(); i++)
        {
            sprintf(szlabel, "%d", i);
            labels_.push_back(szlabel);
        }

    }
}

std::pair<std::string,float> CharsRecognize::recognize(const cv::Mat &im)
{
    cv::Mat im_src;
    cv::resize(im,im_src,cv::Size(110,32));
    cv::Mat im_resized = imResize(im_src);
    cv::Mat im_normalized = imConvert(im_resized);
    wrapInputLayer(im_normalized);

    net_->Forward();

#if 0
    const boost::shared_ptr<Blob<float> > blob = net_->blob_by_name("fc1x");
    int n = blob->num();
    int c = blob->channels();
    int h = blob->height();
    int w = blob->width();
    int size = blob->count();
    cout<<"fc1x"<<" : "<<n<<" "<<c<<" "<<h<<" "<<w<<endl;
    vector<float> out_data;
    vector<int> out_shape;

    getLayerFeatureMaps(string("fc1x"),out_data,out_shape);

    int max_index = 0;
    int t = c*h*w;
    float max_value = 0;
    for(int i=0; i<out_data.size(); ++i)
    {
        if(i%t==0 && i)
        {
            cout<<"max_index: "<<max_index<<"\tmax_value: "<<max_value<<endl;
        }

        if(out_data[i]>max_value)
        {
            max_value = out_data[i];
            max_index = i%t;
        }
    }


    vector<float> result_data;
    vector<int> result_shape;
    getLayerFeatureMaps(string("result"),result_data,result_shape);
#endif

    Blob<float>* output_layer = net_->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->count();
    vector<float> fm = std::vector<float>(begin, end);
    vector<int> predicts;
    string chars = getPredictString(fm,predicts);

    std::pair<string,float> result;
    result.first = chars;

    vector<float> out_data;
    vector<int> out_shape;

    getLayerFeatureMaps(string("fc1x"),out_data,out_shape);
    int tempsteps = out_shape[0];
    float score = getCTCLoss(out_data.data(),tempsteps,predicts);

    result.second = score;
    return result;
}

cv::Mat CharsRecognize::imResize(const cv::Mat &im)
{
    cv::Mat im_src;
    int im_w = im.cols,im_h = im.rows;
    if(2 * im_w < im_h)
    {
        cv::transpose(im,im_src);
        cv::flip(im_src,im_src,1);
        im_w = im_src.cols,im_h = im.rows;
    } else
        im_src = im;

    int hstd = input_geometry_.height;
    int w1 = hstd*im_w/im_h;

    cv::Size size = cv::Size(w1,hstd);
    cv::Mat im_resized;
    cv::resize(im_src,im_resized,size);

    return im_resized;

}

void CharsRecognize::getLayerFeatureMaps(const string &layer_name, vector<float> &out_data, vector<int> &out_shape)
{
    out_data.clear();
    out_shape.clear();

    const boost::shared_ptr<Blob<float> >& blob = net_->blob_by_name(layer_name);
    if (!blob)
        return ;

    const float* begin = blob->cpu_data();
    const float* end = begin + blob->count();
    out_shape = blob->shape();
    out_data = std::vector<float>(begin, end);

}

cv::Mat CharsRecognize::imConvert(const cv::Mat &im)
{
    cv::Mat sample;
    if (im.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(im, sample, cv::COLOR_BGR2GRAY);
    else if (im.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(im, sample, cv::COLOR_BGRA2GRAY);
    else if (im.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(im, sample, cv::COLOR_BGRA2BGR);
    else if (im.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(im, sample, cv::COLOR_GRAY2BGR);
    else
        sample = im;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample.convertTo(sample_float, CV_32FC3);
    else
        sample.convertTo(sample_float, CV_32FC1);

    cv::cvtColor(sample_float,sample_float,cv::COLOR_BGR2RGB);

    cv::Mat sample_normalized;
    if(use_mean_file)
        cv::subtract(sample_float,mean_,sample_normalized);
    else
    {
        cv::Mat mean = cv::Mat(sample_float.size(),imgtype,channel_mean);
        cv::subtract(sample_float,mean,sample_normalized);
    }

    return sample_normalized;
}

void CharsRecognize::wrapInputLayer(const Mat &im)
{
    vector<Mat> input_channels;
    input_channels.clear();
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1,num_channels_,im.rows,im.cols);
    net_->Reshape();

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();

    int size = width * height;
    for (int i = 0; i < num_channels_; ++i)
    {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += size;
    }

    cv::split(im,input_channels);
    assert(reinterpret_cast<float*>(input_channels.at(0).data)==net_->input_blobs()[0]->cpu_data());
}

void CharsRecognize::setMean(const string &mean_file)
{
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
        << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
        /* Extract an individual channel. */
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);
    mean.convertTo(mean_, CV_32FC3);
}

void CharsRecognize::setMean(const vector<float> &mean_values)
{
    assert(mean_values.size()==num_channels_);
    for(int i=0; i<mean_values.size(); ++i)
    {
        channel_mean[i] = mean_values[i];
    }
//
}

string CharsRecognize::getPredictString(const vector<float> &fm, vector<int>& predicts)
{
    predicts.clear();
    string chars;
    for(int i=0; i<fm.size(); ++i)
    {
        int label = int(fm[i]+0.5f);
        if(label>=0 && label != index_blank)
        {
            chars += labels_[label];
            predicts.push_back(label);
        }
    }
    return chars;
}

ICharsRecognize *CreateICharsRecognize(const std::string& model_dir,const int gpu_id)
{
    string temp_dir = model_dir;
    if(temp_dir[temp_dir.size()-1]!='/')
        temp_dir += "/";
    string deploy_file = temp_dir+"plate/recogniser/deploy.prototxt";
    string weight_file = temp_dir+"plate/recogniser/weights.caffemodel";
    string label_file = temp_dir+"plate/recogniser/labels.txt";
    string mean_file = "";
    vector<float> mean_value = {152,152,152};
    ICharsRecognize *recognizer = new CharsRecognize(deploy_file,weight_file,mean_file,mean_value,label_file,gpu_id);

    return recognizer;
}
