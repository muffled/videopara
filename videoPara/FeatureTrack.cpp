#include "FeatureTrack.h"

using namespace std;
using namespace cv;

CLKTTracker::CLKTTracker()
{
    m_maxCorners = 1000;
    m_qualityLevel = 0.01;
    m_minDistance = 3;
}

void CLKTTracker::goodFeaturesToTrack(cv::Mat& img,std::vector<cv::Point2f>& features)
{
    Mat grayMat;
    assert(1 == img.channels() || 3 == img.channels());
    if (1 == img.channels())  img.copyTo(grayMat);
    else
    {
        cvtColor(img,grayMat,CV_BGR2GRAY);
    }

    vector<Point2f> corners;
    cv::goodFeaturesToTrack(grayMat(m_tripwireWindow),corners,m_maxCorners,m_qualityLevel,m_minDistance,Mat(),3);
    Point offPoint(m_tripwireWindow.x,m_tripwireWindow.y);
    for (size_t i = 0;i < corners.size();i++)
    {
        Point2f point(corners[i].x + offPoint.x,corners[i].y + offPoint.y);
        features.push_back(point);
    }
}

void CLKTTracker::goodFeaturesToTrack(cv::Mat& img,std::list<CTrajectory>& listTrajectory,ulong nFrame)
{
    Mat grayMat;
    assert(1 == img.channels() || 3 == img.channels());
    if (1 == img.channels())  img.copyTo(grayMat);
    else
    {
        cvtColor(img,grayMat,CV_BGR2GRAY);
    }

    vector<Point2f> corners;
    cv::goodFeaturesToTrack(grayMat(m_tripwireWindow),corners,m_maxCorners,m_qualityLevel,m_minDistance,Mat(),3);

    // offPoint:���ʱ�������ͼƬ��Ϊԭͼ��tripwire window���ڵĲ��֡�
    //          ����cv::goodFeaturesToTrack ����⵽�������������������tripwire window���ԡ�
    //          ���ڴ洢����ʱ����Ҫ����offPoint������⵽�����껹ԭ��ԭͼ�С�
    Point offPoint(m_tripwireWindow.x,m_tripwireWindow.y);

    for (vector<Point2f>::size_type i = 0;i < corners.size();i++)
    {
        Point2f point(corners[i].x + offPoint.x,corners[i].y + offPoint.y);
        CTrajectory trajectory(point,nFrame);   
        listTrajectory.push_back(trajectory);
    }
}

void CLKTTracker::calcOpticalFlowPyrLK(std::list<CTrajectory>& listTrajectory,std::list<CTrajectory>& finished_listTrajectory,cv::Mat& curFrame,ulong nFrame)
{
    Mat curGrayFrame;
    assert(1 == curFrame.channels() || 3 == curFrame.channels());
    if (1 == curFrame.channels())  curFrame.copyTo(curGrayFrame);
    else
    {
        cvtColor(curFrame,curGrayFrame,CV_BGR2GRAY);
    }

    // ��ȡÿ������������һ֡��λ��
    vector<Point2f> prevCorners;
    for (list<CTrajectory>::iterator it = listTrajectory.begin();it != listTrajectory.end();it++)
    {
        prevCorners.push_back(it->getTrajectoryEnd());
    }

    // �ڵ�ǰ֡�и����������λ��
    vector<Point2f> curCorners;
    vector<uchar> status;
    vector<float> err;
    cv::calcOpticalFlowPyrLK(m_prevFrame,curGrayFrame,prevCorners,curCorners,status,err); 

    // ���������
    size_t i = 0;
    for (list<CTrajectory>::iterator it = listTrajectory.begin();it != listTrajectory.end();i++)
    {
        // ɾ��û��׷�ٵ���������
        if (!status[i])
        {           
            it = listTrajectory.erase(it);  // ������ָ��ɾ��Ԫ�ص���һ��Ԫ��
            continue;
        }

        // ɾ��ͻȻ��Ծ�ĵ�,��Щ��ͨ��Ϊ���
        if (it->getTrajectorySize() > 1)
        {
            size_t j = it->getTrajectorySize();
            float y_diff = (it->getTrajectory())[j-1].y - (it->getTrajectory())[j-2].y;
            float x_diff = (it->getTrajectory())[j-1].x - (it->getTrajectory())[j-2].x;
            float pre_dist = x_diff * x_diff + y_diff * y_diff;

            y_diff = curCorners[i].y - (it->getTrajectory())[j-1].y;
            x_diff = curCorners[i].x - (it->getTrajectory())[j-1].x;
            float cur_dist = x_diff * x_diff + y_diff * y_diff;

            if (cur_dist > 25*pre_dist+1)
            {
                it = listTrajectory.erase(it);
                continue;
            }
        }

        // ������������ڵ�ǰ֡�е�λ��
        it->AddPoint(curCorners[i]);

        // ��������ⴰ�ڵ�������ת���� finished_listTrajectory����־һ���켣���γ�
        if (!m_detectWindow.contains(curCorners[i]))
        {  
            if (curCorners[i].y < m_detectWindow.y &&
                curCorners[i].x >= m_detectWindow.x &&
                curCorners[i].x <= (m_detectWindow.x + m_detectWindow.width) )
            {
                it->setEndFrame(nFrame);   // ��¼�½�����֡��
                finished_listTrajectory.push_back(*it);
            }
            it = listTrajectory.erase(it);
            continue;
        }
        it++;
    }

    // ���� m_prevFrame
    curGrayFrame.copyTo(m_prevFrame);
}

void CLKTTracker::calcOpticalFlowPyrLK(std::list<CTrajectory>& listTrajectory,std::list<CTrajectory>& finished_listTrajectory,
                                       std::list<CTrajectory>& unfinished_listTrajectory,cv::Mat& curFrame,ulong nFrame)
{
    Mat curGrayFrame;
    assert(1 == curFrame.channels() || 3 == curFrame.channels());
    if (1 == curFrame.channels())  curFrame.copyTo(curGrayFrame);
    else
    {
        cvtColor(curFrame,curGrayFrame,CV_BGR2GRAY);
    }

    // ��ȡÿ������������һ֡��λ��
    vector<Point2f> prevCorners;
    for (list<CTrajectory>::iterator it = listTrajectory.begin();it != listTrajectory.end();it++)
    {
        prevCorners.push_back(it->getTrajectoryEnd());
    }

    // �ڵ�ǰ֡�и����������λ��
    vector<Point2f> curCorners;
    vector<uchar> status;
    vector<float> err;
    cv::calcOpticalFlowPyrLK(m_prevFrame,curGrayFrame,prevCorners,curCorners,status,err); 

    // ���������
    size_t i = 0;
    for (list<CTrajectory>::iterator it = listTrajectory.begin();it != listTrajectory.end();i++)
    {
        // ɾ��û��׷�ٵ���������
        if (!status[i])
        {          
            unfinished_listTrajectory.push_back(*it);
            it = listTrajectory.erase(it);  // ������ָ��ɾ��Ԫ�ص���һ��Ԫ��
            continue;
        }

        // ɾ��ͻȻ��Ծ�ĵ�,��Щ��ͨ��Ϊ���
        if (it->getTrajectorySize() > 1)
        {
            size_t j = it->getTrajectorySize();
            float y_diff = (it->getTrajectory())[j-1].y - (it->getTrajectory())[j-2].y;
            float x_diff = (it->getTrajectory())[j-1].x - (it->getTrajectory())[j-2].x;
            float pre_dist = x_diff * x_diff + y_diff * y_diff;

            y_diff = curCorners[i].y - (it->getTrajectory())[j-1].y;
            x_diff = curCorners[i].x - (it->getTrajectory())[j-1].x;
            float cur_dist = x_diff * x_diff + y_diff * y_diff;

            if (cur_dist > 25*pre_dist+1)
            {
				//unfinished_listTrajectory.push_back(*it);
                it = listTrajectory.erase(it);
                continue;
            }
        }

        // ������������ڵ�ǰ֡�е�λ��
        it->AddPoint(curCorners[i]);

        // ��������ⴰ�ڵ�������ת���� finished_listTrajectory����־һ���켣���γ�
        if (!m_detectWindow.contains(curCorners[i]))
        {  
            if (curCorners[i].y < m_detectWindow.y &&
                curCorners[i].x >= m_detectWindow.x &&
                curCorners[i].x <= (m_detectWindow.x + m_detectWindow.width) )
            {
                it->setEndFrame(nFrame);   // ��¼�½�����֡��
                finished_listTrajectory.push_back(*it);
            }
            else
            {
                // ��;�ݳ���ⴰ�ڣ�Ҳ����Ϊ�Ǹ���ʧ�ܡ�
                unfinished_listTrajectory.push_back(*it);
            }
            it = listTrajectory.erase(it);
            continue;
        }
        it++;
    }

    // ���� m_prevFrame
    curGrayFrame.copyTo(m_prevFrame);
}