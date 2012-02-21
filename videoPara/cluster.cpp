#include "cluster.h"

using namespace cv;
using namespace std;

inline float pointDistance(Point& a,Point2f& b)
{
    return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}

void CTrajectoryCluster::cluster(std::list<CTrajectory>& listTrajectory)
{
    vector<clusterPoints>::size_type i = 0;

    // 取出每个轨迹的生命点(开始帧数、结束帧数）
    vector<Point> candidatePoints;
    for (list<CTrajectory>::iterator it = listTrajectory.begin();it != listTrajectory.end();it++)
    {
        Point point = it->getLife();
        candidatePoints.push_back(point);
    }
    if (0 == candidatePoints.size()) return;

    // 增加第一个聚类中心点
    clusterPoints tempCluster;
    tempCluster.m_center = candidatePoints[0];
    tempCluster.m_memberPoints.push_back(candidatePoints[0]);
    m_clusterPoints.push_back(tempCluster);

    for (vector<Point>::size_type i = 1;i < candidatePoints.size();i++)
    {
        // 提取距离候选点最近的聚类中心点
        float minDistance = 1000000000000.f;
        vector<clusterPoints>::size_type index = 0;
        for (vector<clusterPoints>::size_type j = 0;j < m_clusterPoints.size();j++)
        {
            float distance = pointDistance(candidatePoints[i],m_clusterPoints[j].m_center);
            if (distance < minDistance)
            {
                index = j;
                minDistance = distance;
            }
        }

        // 聚类
        float startDistance = (candidatePoints[i].x - m_clusterPoints[index].m_center.x) * m_timePerFrame;
        float endDistance = (candidatePoints[i].y - m_clusterPoints[index].m_center.y) * m_timePerFrame;
        if (abs(startDistance) <= 2.5 * m_varStart && abs(endDistance) <= 2.5 * m_varEnd)
        {
            // 更新聚类点中心的位置
            vector<Point>::size_type clusterSize = m_clusterPoints[index].m_memberPoints.size();
            m_clusterPoints[index].m_center.x += 1.f / (clusterSize + 1) * (candidatePoints[i].x - m_clusterPoints[index].m_center.x);
            m_clusterPoints[index].m_center.y += 1.f / (clusterSize + 1) * (candidatePoints[i].y - m_clusterPoints[index].m_center.y);
            m_clusterPoints[index].m_memberPoints.push_back(candidatePoints[i]);
        }
        else
        {
            tempCluster.m_center = candidatePoints[i];
            tempCluster.m_memberPoints.clear();   // important
            tempCluster.m_memberPoints.push_back(candidatePoints[i]);
            m_clusterPoints.push_back(tempCluster);            
        }        
    }
}

void CTrajectoryCluster::cluster(std::list<CTrajectory>& listTrajectory,int /*param*/)
{
    if (listTrajectory.size() == 0) return;

    list<CTrajectory>::iterator it = listTrajectory.begin();

    // 增加第一个聚类中心点
    clusterTrajectory tempCluster;
    tempCluster.m_center = it->getLife();
    tempCluster.m_memberTrajectory.push_back(*it);
    m_clusterTrajectory.push_back(tempCluster);

    it++;  // 指向下一个轨迹

    for (;it != listTrajectory.end();it++)
    {
        Point lifePoint = it->getLife();
        // 提取距离候选点最近的聚类中心点
        float minDistance = 1000000000000.f;
        vector<clusterTrajectory>::size_type index = 0;
        vector<clusterTrajectory>::size_type size = m_clusterTrajectory.size();
        for (vector<clusterTrajectory>::size_type j = 0;j < m_clusterTrajectory.size();j++)
        {           
            float distance = pointDistance(lifePoint,m_clusterTrajectory[j].m_center);
            if (distance < minDistance)
            {
                index = j;
                minDistance = distance;
            }
        }

        // 聚类
        float startDistance = (lifePoint.x - m_clusterTrajectory[index].m_center.x) * m_timePerFrame;
        float endDistance = (lifePoint.y - m_clusterTrajectory[index].m_center.y) * m_timePerFrame;
        if (abs(startDistance) <= 2.5 * m_varStart && abs(endDistance) <= 2.5 * m_varEnd)
        {
            // 更新聚类点中心的位置          
            vector<CTrajectory>::size_type clusterSize = m_clusterTrajectory[index].m_memberTrajectory.size();
            m_clusterTrajectory[index].m_center.x += 1.f / (clusterSize + 1) * (lifePoint.x - m_clusterTrajectory[index].m_center.x);
            m_clusterTrajectory[index].m_center.y += 1.f / (clusterSize + 1) * (lifePoint.y - m_clusterTrajectory[index].m_center.y);
            m_clusterTrajectory[index].m_memberTrajectory.push_back(*it);
        }
        else
        {
            tempCluster.m_center = lifePoint;
            tempCluster.m_memberTrajectory.clear();   // important
            tempCluster.m_memberTrajectory.push_back(*it);
            m_clusterTrajectory.push_back(tempCluster);            
        }
    }

}