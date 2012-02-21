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

    // ȡ��ÿ���켣��������(��ʼ֡��������֡����
    vector<Point> candidatePoints;
    for (list<CTrajectory>::iterator it = listTrajectory.begin();it != listTrajectory.end();it++)
    {
        Point point = it->getLife();
        candidatePoints.push_back(point);
    }
    if (0 == candidatePoints.size()) return;

    // ���ӵ�һ���������ĵ�
    clusterPoints tempCluster;
    tempCluster.m_center = candidatePoints[0];
    tempCluster.m_memberPoints.push_back(candidatePoints[0]);
    m_clusterPoints.push_back(tempCluster);

    for (vector<Point>::size_type i = 1;i < candidatePoints.size();i++)
    {
        // ��ȡ�����ѡ������ľ������ĵ�
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

        // ����
        float startDistance = (candidatePoints[i].x - m_clusterPoints[index].m_center.x) * m_timePerFrame;
        float endDistance = (candidatePoints[i].y - m_clusterPoints[index].m_center.y) * m_timePerFrame;
        if (abs(startDistance) <= 2.5 * m_varStart && abs(endDistance) <= 2.5 * m_varEnd)
        {
            // ���¾�������ĵ�λ��
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

    // ���ӵ�һ���������ĵ�
    clusterTrajectory tempCluster;
    tempCluster.m_center = it->getLife();
    tempCluster.m_memberTrajectory.push_back(*it);
    m_clusterTrajectory.push_back(tempCluster);

    it++;  // ָ����һ���켣

    for (;it != listTrajectory.end();it++)
    {
        Point lifePoint = it->getLife();
        // ��ȡ�����ѡ������ľ������ĵ�
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

        // ����
        float startDistance = (lifePoint.x - m_clusterTrajectory[index].m_center.x) * m_timePerFrame;
        float endDistance = (lifePoint.y - m_clusterTrajectory[index].m_center.y) * m_timePerFrame;
        if (abs(startDistance) <= 2.5 * m_varStart && abs(endDistance) <= 2.5 * m_varEnd)
        {
            // ���¾�������ĵ�λ��          
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