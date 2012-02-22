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

    // offPoint:检测时所传入的图片，为原图中tripwire window所在的部分。
    //          利用cv::goodFeaturesToTrack 所检测到的特征点坐标是相对于tripwire window而言。
    //          故在存储坐标时，需要加上offPoint，将检测到的坐标还原到原图中。
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

    // 获取每个特征点在上一帧的位置
    vector<Point2f> prevCorners;
    for (list<CTrajectory>::iterator it = listTrajectory.begin();it != listTrajectory.end();it++)
    {
        prevCorners.push_back(it->getTrajectoryEnd());
    }

    // 在当前帧中跟踪特征点的位置
    vector<Point2f> curCorners;
    vector<uchar> status;
    vector<float> err;
    cv::calcOpticalFlowPyrLK(m_prevFrame,curGrayFrame,prevCorners,curCorners,status,err); 

    // 特征点更新
    size_t i = 0;
    for (list<CTrajectory>::iterator it = listTrajectory.begin();it != listTrajectory.end();i++)
    {
        // 删除没有追踪到的特征点
        if (!status[i])
        {           
            it = listTrajectory.erase(it);  // 迭代器指向删除元素的下一个元素
            continue;
        }

        // 删除突然跳跃的点,这些点通常为噪点
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

        // 储存此特征点在当前帧中的位置
        it->AddPoint(curCorners[i]);

        // 将超出检测窗口的特征点转存至 finished_listTrajectory，标志一条轨迹的形成
        if (!m_detectWindow.contains(curCorners[i]))
        {  
            if (curCorners[i].y < m_detectWindow.y &&
                curCorners[i].x >= m_detectWindow.x &&
                curCorners[i].x <= (m_detectWindow.x + m_detectWindow.width) )
            {
                it->setEndFrame(nFrame);   // 记录下结束的帧数
                finished_listTrajectory.push_back(*it);
            }
            it = listTrajectory.erase(it);
            continue;
        }
        it++;
    }

    // 更新 m_prevFrame
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

    // 获取每个特征点在上一帧的位置
    vector<Point2f> prevCorners;
    for (list<CTrajectory>::iterator it = listTrajectory.begin();it != listTrajectory.end();it++)
    {
        prevCorners.push_back(it->getTrajectoryEnd());
    }

    // 在当前帧中跟踪特征点的位置
    vector<Point2f> curCorners;
    vector<uchar> status;
    vector<float> err;
    cv::calcOpticalFlowPyrLK(m_prevFrame,curGrayFrame,prevCorners,curCorners,status,err); 

    // 特征点更新
    size_t i = 0;
    for (list<CTrajectory>::iterator it = listTrajectory.begin();it != listTrajectory.end();i++)
    {
        // 删除没有追踪到的特征点
        if (!status[i])
        {          
            unfinished_listTrajectory.push_back(*it);
            it = listTrajectory.erase(it);  // 迭代器指向删除元素的下一个元素
            continue;
        }

        // 删除突然跳跃的点,这些点通常为噪点
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

        // 储存此特征点在当前帧中的位置
        it->AddPoint(curCorners[i]);

        // 将超出检测窗口的特征点转存至 finished_listTrajectory，标志一条轨迹的形成
        if (!m_detectWindow.contains(curCorners[i]))
        {  
            if (curCorners[i].y < m_detectWindow.y &&
                curCorners[i].x >= m_detectWindow.x &&
                curCorners[i].x <= (m_detectWindow.x + m_detectWindow.width) )
            {
                it->setEndFrame(nFrame);   // 记录下结束的帧数
                finished_listTrajectory.push_back(*it);
            }
            else
            {
                // 中途逸出检测窗口，也被认为是跟踪失败。
                unfinished_listTrajectory.push_back(*it);
            }
            it = listTrajectory.erase(it);
            continue;
        }
        it++;
    }

    // 更新 m_prevFrame
    curGrayFrame.copyTo(m_prevFrame);
}