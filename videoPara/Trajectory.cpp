#include "stdafx.h"
#include "Trajectory.h"

using namespace std;
using namespace cv;

CTrajectory::CTrajectory(cv::Point2f point,int nFrame)
{
    m_ts = m_te = nFrame;
    m_trajectory.push_back(point);
}

