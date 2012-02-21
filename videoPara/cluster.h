#pragma once

#include "stdafx.h"
#include "Trajectory.h"

// ��¼��ÿһ����������ĵ㼰����ɵ�
struct clusterPoints
{
    cv::Point2f m_center;
    std::vector<cv::Point> m_memberPoints;
};

// ��¼��ÿһ����������ĵ㼰��Ӧ�ĳ�Ա�켣
// �˽ṹ���ں����ָ��ÿһ�������ͼ��
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
    int   m_timePerFrame;    // ����֡��������Ӧ��ʱ��
    std::vector<clusterPoints> m_clusterPoints;
    std::vector<clusterTrajectory> m_clusterTrajectory;
public:
    CTrajectoryCluster():m_varStart(0),m_varEnd(0),m_timePerFrame(0) {}
    CTrajectoryCluster(ulong varStart,ulong varEnd):m_varStart(varStart),m_varEnd(varEnd)  {}
    
    inline std::vector<clusterPoints>& getCluster() {return m_clusterPoints;}
    inline std::vector<clusterTrajectory>& getClusterTrajectory() {return m_clusterTrajectory;}
    inline size_t getClusterNumber() {return m_clusterPoints.size();}

    // ��������
    inline void setVarStart(ulong varStart)  {m_varStart = varStart;}
    inline void setVarEnd(ulong varEnd)      {m_varEnd = varEnd;}
    inline void setTimePerFrame(int timePerFrame)  {m_timePerFrame = timePerFrame;}

    void cluster(std::list<CTrajectory>&);

    void cluster(std::list<CTrajectory>&,int);
};