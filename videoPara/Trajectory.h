#pragma once

#include "stdafx.h"

class  CTrajectory
{
private:	
    /*
        每一条轨迹，m_te - m_ts + 1 == m_trajectory.size();
    */
    ulong   m_te;   // 此轨迹开始出现时所在的帧数
    ulong	m_ts;   // 此轨迹结束出现时所在的帧数
    std::vector<cv::Point2f>  m_trajectory;    // 此轨迹生命期间在每一帧中的位置
public:
    CTrajectory():m_te(0),m_ts(0) {}
    CTrajectory(cv::Point2f,int);

    // 设置属性
    inline void setStartFrame(ulong ustart) {m_ts = ustart;}  
    inline void setEndFrame(ulong uend) {m_te = uend;}
    inline void AddPoint(cv::Point2f point) {m_trajectory.push_back(point);}

    // 读取属性
    inline ulong getStartFrame() {return m_ts;}
    inline ulong getEndFrame()   {return m_te;}
    inline cv::Point2f  getTrajectoryStart()  {return m_trajectory.front();}
    inline cv::Point2f	getTrajectoryEnd()    {return m_trajectory.back();}
    std::vector<cv::Point2f>& getTrajectory() {return m_trajectory;}
    inline size_t getTrajectorySize()  {return m_trajectory.size();}
    inline ulong  getLifeTime()   {return m_te - m_ts;}
    inline cv::Point  getLife()   {return cv::Point(m_ts,m_te);}
};

