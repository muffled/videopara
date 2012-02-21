#pragma once

#include "stdafx.h"
#include "Trajectory.h"

class CTracker;

class CLKTTracker
{
private:
    cv::Mat  m_prevFrame;
    cv::Rect m_tripwireWindow;      // �������ⴰ��
    cv::Rect m_detectWindow;        // ������׷�ٴ���

    // cv::goodFeaturesToTrack ����
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
    

    // ����OpenCV cv::goodFeaturesToTrack���������ء�
    // ��ÿһ֡��tripwire window�м�⵽��������洢��������.
    void goodFeaturesToTrack(cv::Mat& img,std::list<CTrajectory>& listTrajectory,ulong nFrame);

    // ����ÿһ֡�м�ⴰ���е������㡣
    void goodFeaturesToTrack(cv::Mat& img,std::vector<cv::Point2f>& features);

    // ����OpenCV cv::calcOpticalFlowPyrLK����������
    void calcOpticalFlowPyrLK(std::list<CTrajectory>& listTrajectory,std::list<CTrajectory>& finished_listTrajectory,cv::Mat& curFrame,ulong nFrame);

    void calcOpticalFlowPyrLK(std::list<CTrajectory>& listTrajectory,std::list<CTrajectory>& finished_listTrajectory,
                              std::list<CTrajectory>& unfinished_listTrajectory,cv::Mat& curFrame,ulong nFrame);
};