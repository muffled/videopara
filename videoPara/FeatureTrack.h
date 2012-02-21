#pragma once

#include "stdafx.h"
#include "Trajectory.h"

class CTracker;

class CLKTTracker
{
private:
    cv::Mat  m_prevFrame;
    cv::Rect m_tripwireWindow;      // 特征点检测窗口
    cv::Rect m_detectWindow;        // 特征点追踪窗口

    // cv::goodFeaturesToTrack 参数
    int m_maxCorners;
    double m_qualityLevel;
    double m_minDistance;

public:
    CLKTTracker();

    void inline SetPrevFrame(cv::Mat& curFrame) 
    {
        assert(1 == curFrame.channels() || 3 == curFrame.channels());
        if (1 == curFrame.channels()) curFrame.copyTo(m_prevFrame);
        else  cvtColor(curFrame,m_prevFrame,CV_BGR2GRAY);
    }
    void inline SetTripwireWindow(cv::Rect& rect) {m_tripwireWindow = rect;}
    void inline SetDetectWindow(cv::Rect& rect) {m_detectWindow = rect;}
    

    // 对于OpenCV cv::goodFeaturesToTrack函数的重载。
    // 将每一帧在tripwire window中检测到的特征点存储至链表中.
    void goodFeaturesToTrack(cv::Mat& img,std::list<CTrajectory>& listTrajectory,ulong nFrame);

    // 检测出每一帧中检测窗口中的特征点。
    void goodFeaturesToTrack(cv::Mat& img,std::vector<cv::Point2f>& features);

    // 对于OpenCV cv::calcOpticalFlowPyrLK函数的重载
    void calcOpticalFlowPyrLK(std::list<CTrajectory>& listTrajectory,std::list<CTrajectory>& finished_listTrajectory,cv::Mat& curFrame,ulong nFrame);

    void calcOpticalFlowPyrLK(std::list<CTrajectory>& listTrajectory,std::list<CTrajectory>& finished_listTrajectory,
                              std::list<CTrajectory>& unfinished_listTrajectory,cv::Mat& curFrame,ulong nFrame);
};