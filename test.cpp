#include <iostream>
#include <ICharsRecognize.h>
#include <string>
#include <time.h>

using namespace std;
using namespace cv;

void getImPath(string& picDir,vector<string>&imPath)
{
    string cmd = "find "+picDir+" -name *.jpg";
    FILE *fp = popen(cmd.c_str(),"r");
    char buffer[512];
    while(1)
    {
        fgets(buffer,sizeof(buffer),fp);
        if(feof(fp))
            break;
        buffer[strlen(buffer)-1] = 0;
        imPath.push_back(string(buffer));
    }
}

int main() {

    vector<float> mean_value = {152,152,152};
    string model_dir = "/home/zqp/install_lib/models/";
    string deploy_file = model_dir+"plate/recogniser/deploy.prototxt";
    string weight_file = model_dir+"plate/recogniser/weights.caffemodel";
    string label_file = model_dir+"plate/recogniser/labels.txt";
    ICharsRecognize *reconizer = CreateICharsRecognize(deploy_file,weight_file,label_file,mean_value);
    string pic_dir = "/home/zqp/pic/testimge";
    vector<string> impath;
    getImPath(pic_dir,impath);

    for(int i=0; i<impath.size(); ++i)
    {
        cv::Mat im = cv::imread(impath[i]);
        clock_t start,end;
        start = clock();
        pair<string,float> result = reconizer->recognize(im);
        end = clock();
        cout<<"cost time: "<<double(end-start)/1000<<" ms"<<endl;

        cout<<"license: "<<result.first<<"\tscore: "<<result.second<<endl;
        imshow("im",im);

       if(waitKey(0)==27)
           break;
    }

    return 0;
}