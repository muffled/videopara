#pragma once

#include "stdafx.h"

class  CTrajectory
{
private:	
    /*
        ÿһ���켣��m_te - m_ts + 1 == m_trajectory.size();
    */
    ulong   m_te;   // �˹켣��ʼ����ʱ���ڵ�֡��
    ulong	m_ts;   // �˹켣��������ʱ���ڵ�֡��
    std::vector<cv::Point2f>  m_trajectory;    // �˹켣�����ڼ���ÿһ֡�е�λ��
public:
    CTrajectory():m_te(0),m_ts(0) {}
    CTrajectory(cv::Point2f,int);

    // ��������
    inline void setStartFrame(ulong ustart) {m_ts = ustart;}  
    inline void setEndFrame(ulong uend) {m_te = uend;}
    inline void AddPoint(cv::Point2f point) {m_trajectory.push_back(point);}

    // ��ȡ����
    inline ulong getStartFrame() {return m_ts;}
    inline ulong getEndFrame()   {return m_te;}
    inline cv::Point2f  getTrajectoryStart()  {return m_trajectory.front();}
    inline cv::Point2f	getTrajectoryEnd()    {return m_trajectory.back();}
    std::vector<cv::Point2f>& getTrajectory() {return m_trajectory;}
    inline size_t getTrajectorySize()  {return m_trajectory.size();}
    inline ulong  getLifeTime()   {return m_te - m_ts;}
    inline cv::Point  getLife()   {return cv::Point(m_ts,m_te);}
};

