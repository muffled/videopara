#pragma once

#include "stdafx.h"
#include "Trajectory.h"

// 记录下每一个聚类的中心点及其组成点
struct clusterPoints
{
    cv::Point2f m_center;
    std::vector<cv::Point> m_memberPoints;
};

// 记录下每一个聚类的中心点及相应的成员轨迹
// 此结构用于后续分割出每一个聚类的图像。
struct clusterTrajectory
{
    cv::Point2f m_center;
    std::vector<CTrajectory> m_memberTrajectory;
};

class CTrajectoryCluster
{
private:
    ulong m_varStart;
    ulong m_varEnd;
    int   m_timePerFrame;    // 根据帧数计算相应的时间
    std::vector<clusterPoints> m_clusterPoints;
    std::vector<clusterTrajectory> m_clusterTrajectory;
public:
    CTrajectoryCluster():m_varStart(0),m_varEnd(0),m_timePerFrame(0) {}
    CTrajectoryCluster(ulong varStart,ulong varEnd):m_varStart(varStart),m_varEnd(varEnd)  {}
    
    inline std::vector<clusterPoints>& getCluster() {return m_clusterPoints;}
    inline std::vector<clusterTrajectory>& getClusterTrajectory() {return m_clusterTrajectory;}
    inline size_t getClusterNumber() {return m_clusterPoints.size();}

    // 设置属性
    inline void setVarStart(ulong varStart)  {m_varStart = varStart;}
    inline void setVarEnd(ulong varEnd)      {m_varEnd = varEnd;}
    inline void setTimePerFrame(int timePerFrame)  {m_timePerFrame = timePerFrame;}

    void cluster(std::list<CTrajectory>&);

    void cluster(std::list<CTrajectory>&,int);
};