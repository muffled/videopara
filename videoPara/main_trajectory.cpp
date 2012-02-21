/**
    此文件用于展示每一条完整的轨迹。
**/

#include <iostream>
#include "FeatureTrack.h"

using namespace std;
using namespace cv;

void showFinishedTrajecctory(Mat& img,list<CTrajectory>& finishedListTrajecctory);

/*
int main()
{
    VideoCapture video(".\\testvideos\\3.avi");
    if (!video.isOpened())
    {
        cerr<<"can not open the video"<<endl;
        return 0;
    }
    
    CLKTTracker lktTracker;   // 特征点的检测及跟踪
    list<CTrajectory> listTrajecctory,finished_listTrajecctory;  // 存储每一条轨迹及结束的轨迹
    Mat Frame;
    int nFrame = 0;

    Rect tripwire(45,134,230,10);
    Rect detectWindow(45,106,230,28);
    lktTracker.SetTripwireWindow(tripwire);
    lktTracker.SetDetectWindow(detectWindow);
    
    // 去掉开始的10帧
    for (int i = 0;i < 10;i++) video >> Frame;

    // 第一帧中初始化各个点。
    {
        video >> Frame;
        if (!Frame.data)
        {
            cerr<<"Frame data is NULL."<<endl;
            return 0;
        }
        lktTracker.SetPrevFrame(Frame);
        lktTracker.goodFeaturesToTrack(Frame,listTrajecctory,nFrame++);
    }

    while (true)
    {
        video >> Frame;
        if (!Frame.data) break;

        // 角点跟踪
        lktTracker.calcOpticalFlowPyrLK(listTrajecctory,finished_listTrajecctory,Frame,nFrame);
        // 角点检测       
        lktTracker.goodFeaturesToTrack(Frame,listTrajecctory,nFrame++);

        rectangle(Frame,tripwire,Scalar(0,0,255));
        rectangle(Frame,detectWindow,Scalar(255,0,0));

        showFinishedTrajecctory(Frame,finished_listTrajecctory);
        imshow("show",Frame);
        if (waitKey(30) == 27) break;
    }

    return 1;
}
*/

void showFinishedTrajecctory(Mat& img,list<CTrajectory>& finishedListTrajecctory)
{
    list<CTrajectory>::iterator it;
    for (it = finishedListTrajecctory.begin();it != finishedListTrajecctory.end();it++)
    {
        vector<Point2f> trajectory = it->getTrajectory();
        for (size_t i = 0;i < trajectory.size();i++)
        {
            circle(img,trajectory[i],1,Scalar(0,255,255),-1);
        }
    }
}