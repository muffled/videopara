#include "videoParams.h"
#include "vani_log.h"
#include <iostream>
#include <fstream>
#include <set>

using namespace std;
using namespace una_videoparams;
using namespace cv;

void GetTripwireRectAndDetectRect(cv::Size& videoSize,cv::Size& detectSize,int tripwireHeight,cv::Rect& tripwireRect,cv::Rect& detectRect);
void start_end_variance(vector<Point>& points,float& startVar,float& endVar,int time_per_frame);
void lifetime_mean_variance(vector<Point>& points,float& lifetimeMean,float& lifetimeVar,int time_per_frame);

static int g_videoIndex = 1;

bool CVideoParams::readInitPara(std::string spath)
{
    ifstream file(spath.c_str());
    if (!file.is_open())
    {
        string slog("be unable to open the specified file:");
        slog += spath;
        LogWriter(slog.c_str());
        return false;
    }

    Json::Reader reader;
    Json::Value root;
    if (!reader.parse(file,root,false))
    {
        string slog("json parse the specified file:");
        slog = slog + spath +" failed.";
        LogWriter(slog.c_str());
        return false;
    }

    m_initParam.m_video_width = root.get("video_width",320).asInt();
    m_initParam.m_video_height = root.get("video_height",240).asInt();

    m_initParam.m_detect_width = root.get("detect_width",230).asInt();
    m_initParam.m_detect_height = root.get("detect_height",28).asInt();

    m_initParam.m_tripwire_height = root.get("tripwire_height",10).asInt();

    m_initParam.m_time_per_frame = root.get("time_per_frame",40).asInt();

    m_initParam.m_life_mean = root.get("life_mean",251).asInt();
    m_initParam.m_life_std = root.get("life_std",150).asInt();

    m_initParam.m_ts_std = root.get("ts_std",364).asInt();
    m_initParam.m_te_std = root.get("te_std",359).asInt();
    m_initParam.m_sampleThresh = root.get("sampleThresh",10).asInt();

    return true;
}

void CVideoParams::writePassengerFlowPara(std::string outpath)
{
    Json::Value root;
    root["start_std"] = m_passengerFlowParam.m_start_std;
    root["end_std"] = m_passengerFlowParam.m_end_std;
    root["life_mean"] = m_passengerFlowParam.m_life_mean;
    root["life_std"] = m_passengerFlowParam.m_life_std;
    root["trajectory_quantity"] = m_passengerFlowParam.m_trajectory_quantity;

    Json::StyledWriter writer;
    std::string sres = writer.write(root);

    ofstream outfile(outpath.c_str());
    outfile<<sres;
}

void CVideoParams::calcParam(std::vector<clusterPoints>& points)
{
    ulong threshold = m_initParam.m_sampleThresh;
    int time_per_frame = m_initParam.m_time_per_frame;

    for (vector<clusterPoints>::size_type i = 0;i < points.size();i++)
    {
        if (points[i].m_memberPoints.size() < threshold) continue;

        float startVar = 0.f,endVar = 0.f;
        float lifeMean = 0.f,lifeVar = 0.f;
        int quantity = 0;

        start_end_variance(points[i].m_memberPoints,startVar,endVar,time_per_frame);
        lifetime_mean_variance(points[i].m_memberPoints,lifeMean,lifeVar,time_per_frame);
        quantity = points[i].m_memberPoints.size();

        m_passengerFlowParam.m_start_std = cvRound(startVar);
        m_passengerFlowParam.m_end_std = cvRound(endVar);
        m_passengerFlowParam.m_life_mean = cvRound(lifeMean);
        m_passengerFlowParam.m_life_std = cvRound(lifeVar);
        m_passengerFlowParam.m_trajectory_quantity = quantity;
    }
}

bool CVideoParams::Initialize(std::string spath)
{
    if (!readInitPara(spath))
    {
        LogWriter("CVideoParams Initialize failed.");
        return false;
    }
    

    // 设置追踪器的参数
    Rect tripwireRect,detectRect;
    Size videoSize(m_initParam.m_video_width,m_initParam.m_video_height);
    Size detectSize(m_initParam.m_detect_width,m_initParam.m_detect_height);
    int tripwireHeight = m_initParam.m_tripwire_height;
    GetTripwireRectAndDetectRect(videoSize,detectSize,tripwireHeight,tripwireRect,detectRect);
    m_tracker.SetTripwireWindow(tripwireRect);
    m_tracker.SetDetectWindow(detectRect);

    // 设置轨迹确认器的参数
    m_validation.setMeanLifeTime(m_initParam.m_life_mean);
    m_validation.setVarLifeTime(m_initParam.m_life_std);
    m_validation.setTimePerFrame(m_initParam.m_time_per_frame);

    // 设置轨迹聚类器的参数
    m_cluster.setVarStart(m_initParam.m_ts_std);
    m_cluster.setVarEnd(m_initParam.m_te_std);
    m_cluster.setTimePerFrame(m_initParam.m_time_per_frame);

    return true;
}

/**
    针对轨迹聚类的结果，从结果中提取出每一帧中所包含的特征点。
**/
void CVideoParams::getFeaturesPerFrame(std::vector<clusterTrajectory>& clusTra)
{
    for (size_t i = 0;i < clusTra.size();i++)
    {
        if (clusTra[i].m_memberTrajectory.size() < m_initParam.m_sampleThresh) continue;
        for (size_t j = 0;j < clusTra[i].m_memberTrajectory.size();j++)
        {
            CTrajectory& trajectoryTmp = clusTra[i].m_memberTrajectory[j];
            ulong start = trajectoryTmp.getStartFrame();
            ulong end = trajectoryTmp.getEndFrame();
            vector<Point2f> features = trajectoryTmp.getTrajectory();
            map<int,vector<Point2f> >::iterator it;    
            for (size_t k = 0;k < features.size();k++)
            {
                it = m_featuresPerFrame.find(k + start);
                if (it != m_featuresPerFrame.end())
                    it->second.push_back(features[k]);
                else
                {
                    vector<Point2f> tmp;
                    tmp.push_back(features[k]);
                    m_featuresPerFrame.insert(std::pair<int,vector<Point2f> >(k + start,tmp));
                }
            }
        }
    }

}

float _pointDistance(const Point2f& a,const Point2f& b)
{
	return sqrt( (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) );
}
/**
    针对轨迹聚类的结果，从结果中提取出每一帧中所包含的特征点。
**/
void CVideoParams::getFeaturesPerFrame(std::vector<clusterTrajectory>& clusTra,int /*temp*/)
{
	ofstream infofile(".\\trajectoryInfo.txt");
    int color = 0;
	int index = 1;
    for (size_t i = 0;i < clusTra.size();i++)
    {
        if (clusTra[i].m_memberTrajectory.size() < m_initParam.m_sampleThresh) continue;
		int red = 0,green = 0,blue = 0;

		// 给每一聚类赋予不同的颜色
// 		switch (color)
// 		{
// 		case 0:
// 			red = 255;
// 			green = 0;
// 			blue = 0;
// 			break;
// 		case 1:
// 			red = 0;
// 			green = 255;
// 			blue = 0;
// 			break;
// 		case  2:
// 			red = 0;
// 			green = 0;
// 			blue = 255;
// 			break;
// 		case 3:
// 			red = 255;
// 			green = 255;
// 			blue = 0;
// 			break;
// 		default:
// 			break;
// 		}
// 		color = (color == 3) ? 0 : (color + 1); 

		// 输出每一聚类中轨迹的时间信息
		infofile<<endl<<"cluster:"<<index++<<endl;
		// 每一聚类中，轨迹的最短时间，最长时间，平均时间。
		//int minTime = 10000000,maxTime = 0,aveTime;

        for (size_t j = 0;j < clusTra[i].m_memberTrajectory.size();j++)
        {
			// 给每一个轨迹赋予不同的颜色
			int red = rand() % 255;
			int green = rand() % 255;
			int blue = rand() % 255;

            CTrajectory& trajectoryTmp = clusTra[i].m_memberTrajectory[j];
            ulong start = trajectoryTmp.getStartFrame();
            ulong end = trajectoryTmp.getEndFrame();
            vector<Point2f> features = trajectoryTmp.getTrajectory();
            map<int,vector<colorPoint> >::iterator it; 
			colorPoint tmp;
			tmp.m_red = red;
			tmp.m_green = green;
			tmp.m_blue = blue;

			/** 输出轨迹的时间信息 start **/			
			int time = (end - start) * 40;
			Point2f startPoint = features[0];
			Point2f endPoint = features[features.size() - 1];
			float distance = _pointDistance(startPoint,endPoint);
			infofile<<"trajectory:"<<j<<",speed:";
			infofile<<distance * 1000 / time<<endl;
			/** 输出轨迹的时间信息 end   **/

            for (size_t k = 0;k < features.size();k++)
            {
				tmp.m_point = features[k];
                it = m_featuresPerFrame_color.find(k + start);
                if (it != m_featuresPerFrame_color.end())					
					it->second.push_back(tmp);
                else
                {
					vector<colorPoint> colorPointVec;
					colorPointVec.push_back(tmp);
                    m_featuresPerFrame_color.insert(std::pair<int,vector<colorPoint> >(k + start,colorPointVec));                                                        
                }
            }           
        }
    }

}

bool _pointSortCompare(const Point2f& a,const Point2f& b)
{
    return (a.x == b.x) ? (a.y < b.y) : (a.x < b.x);
}


// 用于set中元素的比较
struct _pointCompare
{
    bool operator () (const Point2f a,const Point2f b) const
    {
        if (a.x == b.x)
            return a.y < b.y;
        else
            return a.x < b.x;
    }
};



void CVideoParams::validFeaturePoints(std::string videopath,std::string resultpath)
{
    playVideo(videopath);

    // 轨迹的验证
    for (list<CTrajectory>::iterator it = m_list_finishedTrajectory.begin();it != m_list_finishedTrajectory.end();)
    {
        if (!m_validation.validation(*it)) 
        {
            it = m_list_finishedTrajectory.erase(it);
            continue;
        }
        it++;
    }

    /**
    此处的实现方法为：
    1.通过验证的轨迹即为有效的轨迹。
    2.将每一帧中有效的初始检测点保存至字典中。
    保存方法为：map(帧数，set(初始检测的特征点))。
    3.重新播放视频，对于每一帧在初始窗口检测特征点，判断所检测到的每一特征点是否在字典对应的帧数中存在。
      如果不存在，则判断此点为无效的特征点。
      将有效特征点与无效特征点以不同的颜色显示。
    **/
    map<ulong,set<Point2f,_pointCompare> > validFeaturesPerFrame;
    map<ulong,set<Point2f,_pointCompare> >::iterator mapit;
    list<CTrajectory>::iterator listit;
    for (listit = m_list_finishedTrajectory.begin();listit != m_list_finishedTrajectory.end();listit++)
    {
        ulong nframe = listit->getStartFrame();
        vector<Point2f> trajectory = listit->getTrajectory();
        mapit = validFeaturesPerFrame.find(nframe);
        if (mapit != validFeaturesPerFrame.end())
            mapit->second.insert(trajectory[0]);
        else
        {       
            set<Point2f,_pointCompare> tmp;
            tmp.insert(trajectory[0]);
            validFeaturesPerFrame.insert(std::pair<int,set<Point2f,_pointCompare> >(nframe,tmp));
        }
    }

    // 重新播放视频
    VideoCapture video(videopath);
    if (!video.isOpened())  
    {
        cerr<<"can not open the video"<<endl;
        return;
    }
    Mat Frame;
    int nFrame = 0,index = 0;
    vector<Point2f> features; 
    char szpath[256];
    memset(szpath,0,sizeof(szpath));
    while (true)
    {
        video >> Frame;
        if (!Frame.data) break;

        features.clear();
        m_tracker.goodFeaturesToTrack(Frame,features);

        rectangle(Frame,Rect(45,106,230,28),CV_RGB(0,0,0),1);
        rectangle(Frame,Rect(45,134,230,10),CV_RGB(0,0,0),1);

        int totalFeaturePoints = (int)features.size();
        char szcount[50];
        memset(szcount,0,sizeof(szcount));
        mapit = validFeaturesPerFrame.find(nFrame - 10);
        if (mapit != validFeaturesPerFrame.end())
        {
            set<Point2f,_pointCompare>::iterator setit;           
            for (size_t i = 0;i < features.size();i++)
            {
                setit = mapit->second.find(features[i]);
                if (setit != mapit->second.end())
                {
                    circle(Frame,features[i],1,CV_RGB(255,255,0),-1);                  
                }
                else
                    circle(Frame,features[i],1,CV_RGB(255,0,0),-1);
            }
 
            // 在图片显示出有效点和所有特征点的数量
            sprintf_s(szcount,sizeof(szcount),"valid:%d",mapit->second.size());
            cv::putText(Frame,szcount,Point(10,30),FONT_HERSHEY_PLAIN,1,CV_RGB(255,255,0));
            sprintf_s(szcount,sizeof(szcount),"total:%d",totalFeaturePoints);
            cv::putText(Frame,szcount,Point(10,50),FONT_HERSHEY_PLAIN,1,CV_RGB(255,255,0));
            sprintf_s(szpath,sizeof(szpath),"%s\\%d.jpg",resultpath.c_str(),index++);
            imwrite(szpath,Frame);
        }
        else
        {
            for (size_t i = 0;i < features.size();i++)
            {
                circle(Frame,features[i],2,CV_RGB(255,0,0),-1);
            }

            // 在图片显示出有效点和所有特征点的数量
            sprintf_s(szcount,sizeof(szcount),"valid:%d",0);
            cv::putText(Frame,szcount,Point(10,30),FONT_HERSHEY_PLAIN,1,CV_RGB(255,255,0));
            sprintf_s(szcount,sizeof(szcount),"total:%d",totalFeaturePoints);
            cv::putText(Frame,szcount,Point(10,50),FONT_HERSHEY_PLAIN,1,CV_RGB(255,255,0));
            sprintf_s(szpath,sizeof(szpath),"%s\\%d.jpg",resultpath.c_str(),index++);
            imwrite(szpath,Frame);
        }
        nFrame++;
        imshow("show",Frame);
        if (waitKey(30) == 27) break;
    }
}

void CVideoParams::playVideo(std::string videopath)
{
    VideoCapture video(videopath);
    if (!video.isOpened())
    {
        string stemp("be unable to open the specified video:");
        stemp += videopath;
        LogWriter(stemp.c_str());
        return;
    }

    m_videoPath = videopath;

    // 处理第二个视频时，将前一个视频的结果清空
    m_listTrajectory.clear();
    m_list_finishedTrajectory.clear();
    m_list_unfinishedTrajectory.clear();

    Mat Frame;
    int nFrame = 0;
    for (size_t i = 0;i < 10;i++) video>>Frame;   // 去掉开始的10帧图片

    // 在第一帧中初始化特征点
    {
        video>>Frame;
        if (!Frame.data) 
        {
            LogWriter("Frame data is NULL.");
            return;  
        }
        m_tracker.SetPrevFrame(Frame);

        m_tracker.goodFeaturesToTrack(Frame,m_listTrajectory,nFrame++);       
    }

    namedWindow("show");

    while (true)
    {
        video>>Frame;
        if (!Frame.data) break;

        // 角点跟踪
        //m_tracker.calcOpticalFlowPyrLK(m_listTrajectory,m_list_finishedTrajectory,Frame,nFrame);
        m_tracker.calcOpticalFlowPyrLK(m_listTrajectory,m_list_finishedTrajectory,
                                       m_list_unfinishedTrajectory,Frame,nFrame);
        // 角点检测       
        m_tracker.goodFeaturesToTrack(Frame,m_listTrajectory,nFrame++);

        rectangle(Frame,Rect(45,106,230,28),CV_RGB(255,255,0),2);
        rectangle(Frame,Rect(45,134,230,10),CV_RGB(255,0,0),2);
//         rectangle(Frame,Rect(90,212,460,56),CV_RGB(255,255,0),1);
//         rectangle(Frame,Rect(90,268,460,20),CV_RGB(255,0,0),1);


        imshow("show",Frame);
        if (waitKey(30) > 0) break;
    }
}

void CVideoParams::videoProcess(std::string videopath,std::string resultpath)
{
    VideoCapture video(videopath);
    if (!video.isOpened())
    {
        string stemp("be unable to open the specified video:");
        stemp += videopath;
        LogWriter(stemp.c_str());
        return;
    }

    m_videoPath = videopath;

    // 处理第二个视频时，将前一个视频的结果清空
    m_listTrajectory.clear();
    m_list_finishedTrajectory.clear();

    Mat Frame;
    int nFrame = 0;
    for (size_t i = 0;i < 10;i++) video>>Frame;   // 去掉开始的10帧图片

    // 在第一帧中初始化特征点
    {
        video>>Frame;
        if (!Frame.data) 
        {
            LogWriter("Frame data is NULL.");
            return;  
        }
        m_tracker.SetPrevFrame(Frame);
        m_tracker.goodFeaturesToTrack(Frame,m_listTrajectory,nFrame++);       
    }
    
    namedWindow("show");

    while (true)
    {
        video>>Frame;
        if (!Frame.data) break;
        
        // 角点跟踪
        m_tracker.calcOpticalFlowPyrLK(m_listTrajectory,m_list_finishedTrajectory,Frame,nFrame);
        // 角点检测       
        m_tracker.goodFeaturesToTrack(Frame,m_listTrajectory,nFrame++);
      
        rectangle(Frame,Rect(45,106,230,28),CV_RGB(255,255,0),2);
        rectangle(Frame,Rect(45,134,230,10),CV_RGB(255,0,0),2);
// 		rectangle(Frame,Rect(90,212,460,56),CV_RGB(255,255,0),1);
// 		rectangle(Frame,Rect(90,268,460,20),CV_RGB(255,0,0),1);

        imshow("show",Frame);
        if (waitKey(30) > 0) break;
    }

    // 轨迹的验证
    for (list<CTrajectory>::iterator it = m_list_finishedTrajectory.begin();it != m_list_finishedTrajectory.end();)
    {
        if (!m_validation.validation(*it)) 
        {
            it = m_list_finishedTrajectory.erase(it);
            continue;
        }
        it++;
    }

    // 轨迹的聚类
    m_cluster.cluster(m_list_finishedTrajectory);

    // 轨迹聚类，用于分割出某一类轨迹的图像
    /****************************************************/
    int ntemp = 0;
    m_cluster.cluster(m_list_finishedTrajectory,ntemp);
    vector<clusterTrajectory> clusTra = m_cluster.getClusterTrajectory();
    //getFeaturesPerFrame(clusTra);
    getFeaturesPerFrame(clusTra,1);

    video.set(CV_CAP_PROP_POS_FRAMES,0);
    nFrame = 0;
    map<int,vector<colorPoint> >::iterator it;
    char szpath[256];
    memset(szpath,0,sizeof(szpath));
    int index = 0;
    while (true)
    {
        video >> Frame;
        if (!Frame.data) break;
        rectangle(Frame,Rect(45,106,230,28),CV_RGB(0,0,0),1);
        rectangle(Frame,Rect(45,134,230,10),CV_RGB(0,0,0),1);
// 		rectangle(Frame,Rect(90,212,460,56),CV_RGB(0,0,0),1);
// 		rectangle(Frame,Rect(90,268,460,20),CV_RGB(0,0,0),1);
        it = m_featuresPerFrame_color.find(nFrame - 10);
        if (it != m_featuresPerFrame_color.end())
        {
            vector<colorPoint> tmp = it->second;
            for (size_t i = 0;i < tmp.size();i++)
            {
				int red = tmp[i].m_red;
				int green = tmp[i].m_green;
				int blue = tmp[i].m_blue;
                circle(Frame,tmp[i].m_point,2,CV_RGB(red,green,blue),-1);
            }
            sprintf_s(szpath,sizeof(szpath),"%s\\%d.jpg",resultpath.c_str(),index++);
            imwrite(szpath,Frame);
        }
        imshow("show",Frame);
        if (waitKey(30) == 27) break;
        nFrame++;
    }


/*
    char szpath[20];
    memset(szpath,0,sizeof(szpath));
    for (vector<clusterTrajectory>::size_type i = 0;i < clusTra.size();i++)
    {
        if (clusTra[i].m_memberTrajectory.size() < m_initParam.m_sampleThresh) continue;

        // 获取聚类中心处于视频的第几帧
        unsigned int nFrame = cvRound( (clusTra[i].m_center.y - clusTra[i].m_center.x) / 2  + clusTra[i].m_center.x) + 10; 
        cout<<"nFrame is:"<<nFrame<<endl;

        vector<Point2f> points;
        vector<Point2f> centerPoints;  // 储存聚类中心的那一帧中所有轨迹特征点的位置。
        for (vector<CTrajectory>::size_type j = 0;j < clusTra[i].m_memberTrajectory.size();j++)
        {
            vector<Point2f> traPoints = clusTra[i].m_memberTrajectory[j].getTrajectory();
            bool is_saved = false;  // 记录聚类中心的那一帧中轨迹特征点的位置是否已经存储
            int nstartFrame = clusTra[i].m_memberTrajectory[j].getStartFrame();
            //cout<<"nstartFrame is:"<<nstartFrame<<endl;
            for (vector<Point2f>::size_type k = 0;k < traPoints.size();k++)
            {
                if (!is_saved && (nstartFrame + k + 10) >= nFrame)
                {
                    centerPoints.push_back(traPoints[k]);
                    is_saved = true;
                }
//                 if (0 == k || k == traPoints.size() - 1)
//                     centerPoints.push_back(traPoints[k]);
                points.push_back(traPoints[k]);
            }
        }
        Mat pointMat(points);
        Rect rect = boundingRect(pointMat);
        cout<<"rect is:"<<rect.x<<","<<rect.y<<","<<rect.width<<","<<rect.height<<endl;
        
        video.set(CV_CAP_PROP_POS_FRAMES,nFrame);
        Mat tempFrame;
        video>>tempFrame;
        sprintf(szpath,".\\img2\\%d.jpg",g_videoIndex++);      
        cout<<"centerPoints size is:"<<centerPoints.size()<<endl;
        rectangle(tempFrame,Rect(45,106,230,28),CV_RGB(0,0,0),1);
        rectangle(tempFrame,Rect(45,134,230,10),CV_RGB(0,0,0),1);
        for (size_t i = 0;i < centerPoints.size();i++)
        {
            circle(tempFrame,centerPoints[i],1,CV_RGB(255,255,0),-1);
        }
        imwrite(szpath,tempFrame);
    }
 */   
    /****************************************************/
/*
    vector<clusterPoints> clusters = m_cluster.getCluster();

//#ifdef _DEBUG
    cout<<"cluster size is:"<<m_cluster.getClusterNumber()<<endl;
    for (vector<clusterPoints>::size_type i = 0;i < clusters.size();i++)
    {
        g_outfile<<"cluster "<<i<<" :"<<clusters[i].m_center.x<<","<<clusters[i].m_center.y<<endl;
        for (vector<Point>::size_type j = 0;j < clusters[i].m_memberPoints.size();j++)
        {
            g_outfile<<clusters[i].m_memberPoints[j].x<<","<<clusters[i].m_memberPoints[j].y<<endl;
        }
    }
//#endif

    calcParam(clusters);   // 计算参数
    writePassengerFlowPara(resultpath);

*/
}


void CVideoParams::featuresDistribute(std::string videopath,std::string resultpath)
{
	playVideo(videopath);

	// 轨迹的验证
	for (list<CTrajectory>::iterator it = m_list_finishedTrajectory.begin();it != m_list_finishedTrajectory.end();)
	{
		if (!m_validation.validation(*it)) 
		{
			it = m_list_finishedTrajectory.erase(it);
			continue;
		}
		it++;
	}

	// 轨迹的聚类
	m_cluster.cluster(m_list_finishedTrajectory,1);
	vector<clusterTrajectory> clusTra = m_cluster.getClusterTrajectory();

	// 输出每一聚类中轨迹中信息。
	string filename = resultpath + ".\\trajectoryInfo.txt";
	ofstream infofile(filename.c_str());

	map<int,vector<colorPoint> > featuresPerFrame_color;  // 用于展示行人上车时，每一帧的特征点分布
	map<int,vector<colorPoint> >::iterator it; 
	map<int,vector<clusterInfo> > clusterInfo_map;   // 用于展示每一聚类的开始帧和结束帧，并在结束帧上输出信息
	map<int,vector<clusterInfo> >::iterator clusInfo_it;
	int validClusterIndex = 1;
	int color = 0;
	for (size_t i = 0;i < clusTra.size();i++)
	{
		if (clusTra[i].m_memberTrajectory.size() < m_initParam.m_sampleThresh) continue;

		int red = 0,blue = 0,green = 0;
		// 给每一聚类赋予不同的颜色
		switch (color)
		{
		case 0:
			red = 255;
			green = 0;
			blue = 0;
			break;
		case 1:
			red = 0;
			green = 255;
			blue = 0;
			break;
		case  2:
			red = 0;
			green = 0;
			blue = 255;
			break;
		case 3:
			red = 255;
			green = 255;
			blue = 0;
			break;
		default:
			break;
		}
		color = (color == 3) ? 0 : (color + 1); 

		// 输出每一聚类中轨迹的时间信息
		if (validClusterIndex > 1) infofile<<endl;
		infofile<<"cluster:"<<validClusterIndex++<<endl;	
		float aveSpeed = 0.f;
		int clusterStartFrame = 100000,clusterEndFrame = 0;
				
		for (size_t j = 0;j < clusTra[i].m_memberTrajectory.size();j++)
		{
			CTrajectory& trajectoryTmp = clusTra[i].m_memberTrajectory[j];
			ulong start = trajectoryTmp.getStartFrame();  
			ulong end = trajectoryTmp.getEndFrame();      
			vector<Point2f> features = trajectoryTmp.getTrajectory();
			

			// 每一轨迹赋予不同的颜色
			colorPoint tmp;
			tmp.m_red = red;
			tmp.m_green = green;
			tmp.m_blue = blue;

			/** 输出轨迹的速度信息 start **/			
			int time = (end - start) * 40;
			Point2f startPoint = features[0];
			Point2f endPoint = features[features.size() - 1];
			float distance = _pointDistance(startPoint,endPoint);
			float speed = distance * 1000 / time;
			infofile<<"trajectory:"<<j<<",speed:";
			infofile<<speed<<endl;

			aveSpeed += speed;
			if (clusterStartFrame > (int)start) clusterStartFrame = start;
			if (clusterEndFrame < (int)end) clusterEndFrame = end;
			/** 输出轨迹的时间信息 end   **/

			for (size_t k = 0;k < features.size();k++)
			{
				tmp.m_point = features[k];
				it = featuresPerFrame_color.find(k + start);
				if (it != featuresPerFrame_color.end())					
					it->second.push_back(tmp);
				else
				{
					vector<colorPoint> colorPointVec;
					colorPointVec.push_back(tmp);
					featuresPerFrame_color.insert(std::pair<int,vector<colorPoint> >(k + start,colorPointVec));                                                        
				}
			}           
		}

		clusterInfo clusinfo;
		clusinfo.startFrame = clusterStartFrame;
		clusinfo.endFrame = clusterEndFrame;
		clusinfo.totalTrajectorys = clusTra[i].m_memberTrajectory.size(); 
		clusinfo.aveSpeed = aveSpeed / clusTra[i].m_memberTrajectory.size();
		
		// 起始帧
		clusInfo_it = clusterInfo_map.find(clusterStartFrame);
		if (clusInfo_it != clusterInfo_map.end())
			clusInfo_it->second.push_back(clusinfo);
		else
		{
			vector<clusterInfo> tmp;
			tmp.push_back(clusinfo);
			clusterInfo_map.insert(std::pair<int,vector<clusterInfo> >(clusterStartFrame,tmp));
		}
		// 结束帧
		clusInfo_it = clusterInfo_map.find(clusterEndFrame);
		if (clusInfo_it != clusterInfo_map.end())
			clusInfo_it->second.push_back(clusinfo);
		else
		{
			vector<clusterInfo> tmp;
			tmp.push_back(clusinfo);
			clusterInfo_map.insert(std::pair<int,vector<clusterInfo> >(clusterEndFrame,tmp));
		}
	}

	/**** 第二次播放，输出相应的信息  *****/
	VideoCapture video(videopath);
	char szpath[256];
	memset(szpath,0,sizeof(szpath));
	int index = 0,nFrame = 0;
	Mat Frame;
	while (true)
	{
		video >> Frame;
		if (!Frame.data) break;

		rectangle(Frame,Rect(45,106,230,28),CV_RGB(0,0,0),1);
		rectangle(Frame,Rect(45,134,230,10),CV_RGB(0,0,0),1);

		// 展示每一聚类起始与结束帧的信息
		clusInfo_it = clusterInfo_map.find(nFrame - 10);
		if (clusInfo_it != clusterInfo_map.end())
		{
			vector<clusterInfo>& tmp = clusInfo_it->second;
			if ( (nFrame - 10) == tmp[0].startFrame)
			{
				const char* szinfo = "cluster start";
				cv::putText(Frame,szinfo,Point(10,40),FONT_HERSHEY_PLAIN,1,CV_RGB(255,255,0));
			}
			if ( (nFrame - 10) == tmp[0].endFrame)
			{
				char szinfo[256];
				memset(szinfo,0,sizeof(szinfo));
				sprintf_s(szinfo,sizeof(szinfo),"total tras:%d",tmp[0].totalTrajectorys);
				cv::putText(Frame,szinfo,Point(10,40),FONT_HERSHEY_PLAIN,1,CV_RGB(255,255,0));
				memset(szinfo,0,sizeof(szinfo));
				sprintf_s(szinfo,sizeof(szinfo),"total frames:%d",tmp[0].endFrame - tmp[0].startFrame + 1);
				cv::putText(Frame,szinfo,Point(10,60),FONT_HERSHEY_PLAIN,1,CV_RGB(255,255,0));
				memset(szinfo,0,sizeof(szinfo));
				sprintf_s(szinfo,sizeof(szinfo),"ave speed:%f",tmp[0].aveSpeed);
				cv::putText(Frame,szinfo,Point(10,80),FONT_HERSHEY_PLAIN,1,CV_RGB(255,255,0));
			}

			// 可能出现某一聚类的结束帧恰好是另一聚类开始帧的情况
			if (tmp.size() > 1)
			{
				const char* szinfo = "cluster start";
				cv::putText(Frame,szinfo,Point(10,80),FONT_HERSHEY_PLAIN,1,CV_RGB(255,255,0));				
			}
		}

		// 展示特征点的分布
		it = featuresPerFrame_color.find(nFrame - 10);
		if (it != featuresPerFrame_color.end())
		{
			vector<colorPoint>& tmp = it->second;
			for (size_t i = 0;i < tmp.size();i++)
			{
				int red = tmp[i].m_red;
				int green = tmp[i].m_green;
				int blue = tmp[i].m_blue;
				circle(Frame,tmp[i].m_point,2,CV_RGB(red,green,blue),-1);
			}
			sprintf_s(szpath,sizeof(szpath),"%s\\%d.jpg",resultpath.c_str(),index++);
			imwrite(szpath,Frame);
		}
		imshow("show",Frame);
		if (waitKey(30) == 27) break;
		nFrame++;
	}
}

void CVideoParams::trackFailurePoints(std::string videopath,std::string resultpath)
{
    VideoCapture video(videopath);
    if (!video.isOpened())
    {
        string stemp("be unable to open the specified video:");
        stemp += videopath;
        LogWriter(stemp.c_str());
        return;
    }

    m_videoPath = videopath;

    // 处理第二个视频时，将前一个视频的结果清空
    m_listTrajectory.clear();
    m_list_finishedTrajectory.clear();
    m_list_unfinishedTrajectory.clear();

    Mat Frame;
    int nFrame = 0;
    for (size_t i = 0;i < 10;i++) video>>Frame;   // 去掉开始的10帧图片

    // 在第一帧中初始化特征点
    {
        video>>Frame;
        if (!Frame.data) 
        {
            LogWriter("Frame data is NULL.");
            return;  
        }
        m_tracker.SetPrevFrame(Frame);
        m_tracker.goodFeaturesToTrack(Frame,m_listTrajectory,nFrame++);       
    }

    namedWindow("show");

    while (true)
    {
        video>>Frame;
        if (!Frame.data) break;

        // 角点跟踪
        m_tracker.calcOpticalFlowPyrLK(m_listTrajectory,m_list_finishedTrajectory,
                                        m_list_unfinishedTrajectory,Frame,nFrame);
        // 角点检测       
        m_tracker.goodFeaturesToTrack(Frame,m_listTrajectory,nFrame++);

        rectangle(Frame,Rect(45,106,230,28),CV_RGB(255,255,0),2);
        rectangle(Frame,Rect(45,134,230,10),CV_RGB(255,0,0),2);

        imshow("show",Frame);
        if (waitKey(30) == 27) break;
    }    

    /**
    将追踪失败的轨迹的所有点提取出来，储存至map中，并显示出来。
    **/
    map<ulong,vector<Point2f> > trackFailurePerFrame;
    map<ulong,vector<Point2f> >::iterator mapit;
    for (list<CTrajectory>::iterator it = m_list_unfinishedTrajectory.begin();
        it != m_list_unfinishedTrajectory.end();it++)
    {
		ulong start = it->getStartFrame();       
		vector<Point2f> features = it->getTrajectory();
		for (size_t i = 0;i < features.size();i++)
		{
			mapit = trackFailurePerFrame.find(start + i);
			if (mapit != trackFailurePerFrame.end())
				mapit->second.push_back(features[i]);
			else
			{
				vector<Point2f> tmp;
				tmp.push_back(features[i]);
				trackFailurePerFrame.insert(pair<int,vector<Point2f> >(i + start,tmp));
			}
		}
    }

    video.set(CV_CAP_PROP_POS_FRAMES,0);
    int index = 0;  
    char szpath[50];
    memset(szpath,0,sizeof(szpath));
    nFrame = 0;
    while (true)
    {
        video >> Frame;
        if (!Frame.data) break;

        rectangle(Frame,Rect(45,106,230,28),CV_RGB(0,0,0),1);
        rectangle(Frame,Rect(45,134,230,10),CV_RGB(0,0,0),1);

        mapit = trackFailurePerFrame.find(nFrame - 10);
        if (mapit != trackFailurePerFrame.end())
        {
            const vector<Point2f>& tmp = mapit->second;
            for (size_t i = 0;i < tmp.size();i++)
            {
                circle(Frame,tmp[i],2,CV_RGB(255,255,0),-1);
            }
            sprintf_s(szpath,sizeof(szpath),"%s\\%d.jpg",resultpath.c_str(),index++);
            imwrite(szpath,Frame);
        }
        imshow("show",Frame);
        if (waitKey(30) == 27) break;
        nFrame++;
    }
}

void CVideoParams::invalidTrajectorys(std::string videopath,std::string resultpath)
{
	playVideo(videopath);

	// 轨迹的验证
	list<CTrajectory> list_invalid_trajectorys;
	for (list<CTrajectory>::iterator it = m_list_finishedTrajectory.begin();it != m_list_finishedTrajectory.end();)
	{
		if (!m_validation.validation(*it)) 
		{
			list_invalid_trajectorys.push_back(*it);
			it = m_list_finishedTrajectory.erase(it);
			continue;
		}
		it++;
	}

	map<int,vector<Point2f> > invalidFeaturesPerFrame;
	map<int,vector<Point2f> >::iterator mapit;
	for (list<CTrajectory>::iterator it = list_invalid_trajectorys.begin();it != list_invalid_trajectorys.end();it++)
	{
		ulong start = it->getStartFrame();  		    
		vector<Point2f> features = it->getTrajectory();
		for (size_t i = 0;i < features.size();i++)
		{
			mapit = invalidFeaturesPerFrame.find(start + i);
			if (mapit != invalidFeaturesPerFrame.end())
				mapit->second.push_back(features[i]);
			else
			{
				vector<Point2f> tmp;
				tmp.push_back(features[i]);
				invalidFeaturesPerFrame.insert(pair<int,vector<Point2f> >(i + start,tmp));
			}
		}
	}

	// 第二次播放显示出视频中无效的轨迹。
	VideoCapture video(videopath);
	int index = 0,nFrame = 0;  
	char szpath[256];
	memset(szpath,0,sizeof(szpath));	
	Mat Frame;
	while (true)
	{
		video >> Frame;
		if (!Frame.data) break;
		rectangle(Frame,Rect(45,106,230,28),CV_RGB(0,0,0),1);
		rectangle(Frame,Rect(45,134,230,10),CV_RGB(0,0,0),1);
		mapit = invalidFeaturesPerFrame.find(nFrame - 10);
		if (mapit != invalidFeaturesPerFrame.end())
		{
			vector<Point2f> tmp = mapit->second;
			for (size_t i = 0;i < tmp.size();i++)
			{
				circle(Frame,tmp[i],2,CV_RGB(255,255,0),-1);
			}
			sprintf_s(szpath,sizeof(szpath),"%s\\%d.jpg",resultpath.c_str(),index++);
			imwrite(szpath,Frame);
		}
		imshow("show",Frame);
		if (waitKey(30) == 27) break;
		nFrame++;
	}
}


ofstream& operator << (ofstream& os,Point2f a)
{
    os<<"("<<a.x<<","<<a.y<<")";
    return os;
}

struct _trajectory
{
    vector<CTrajectory> m_validTrajectory;
    vector<CTrajectory> m_invalidTrajectory;
};

void CVideoParams::featuresStatus(std::string videopath,std::string resultpath)
{
    playVideo(videopath);

    // 轨迹的验证
    list<CTrajectory> list_invalid_trajectorys;
    for (list<CTrajectory>::iterator it = m_list_finishedTrajectory.begin();it != m_list_finishedTrajectory.end();)
    {
        if (!m_validation.validation(*it)) 
        {
            list_invalid_trajectorys.push_back(*it);
            it = m_list_finishedTrajectory.erase(it);
            continue;
        }
        it++;
    }

    m_cluster.cluster(m_list_finishedTrajectory,1);
    vector<clusterTrajectory> clusTra = m_cluster.getClusterTrajectory(); 

    // 将有效轨迹以帧作为键值存放到字典中，不同聚类分开存放。
    vector<map<ulong,vector<CTrajectory> > > clusTra_vec;
    for (size_t i = 0;i < clusTra.size();i++)
    {
        vector<CTrajectory>& travec = clusTra[i].m_memberTrajectory;
        if (travec.size() < m_initParam.m_sampleThresh) continue;

        map<ulong,vector<CTrajectory> > tra_map;
        map<ulong,vector<CTrajectory> >::iterator it;
        for (size_t j = 0;j < travec.size();j++)
        {
            ulong startFrame = travec[j].getStartFrame(); 
            it = tra_map.find(startFrame);
            if (it != tra_map.end())
                it->second.push_back(travec[j]);
            else
            {
                vector<CTrajectory> tmp;
                tmp.push_back(travec[j]);
                tra_map.insert(std::pair<ulong,vector<CTrajectory> >(startFrame,tmp));
            }
        }
        clusTra_vec.push_back(tra_map);
    }

    // 将跟踪丢失的轨迹以帧作为键值存放到字典中。
    map<ulong,vector<CTrajectory> > invalidTra_map;
    map<ulong,vector<CTrajectory> >::iterator invalidTra_map_it;
    for (list<CTrajectory>::iterator it = m_list_unfinishedTrajectory.begin();
        it != m_list_unfinishedTrajectory.end();it++)
    {
        ulong startFrame = it->getStartFrame();
        invalidTra_map_it = invalidTra_map.find(startFrame);
        if (invalidTra_map_it != invalidTra_map.end())
            invalidTra_map_it->second.push_back(*it);
        else
        {
            vector<CTrajectory> tmp;
            tmp.push_back(*it);
            invalidTra_map.insert(std::pair<ulong,vector<CTrajectory> >(startFrame,tmp));
        }            
    } 

    // 将跟踪丢失的轨迹和同其同一帧开始的有效轨迹存储到一起。
    // 轨迹按照聚类分开存储，同一聚类的轨迹按组存储。
    vector<map<ulong,_trajectory> > trajectory_team_cluseter;
    for (size_t i = 0;i < clusTra_vec.size();i++)
    {
        map<ulong,vector<CTrajectory> >& validTra_map = clusTra_vec[i];
        map<ulong,vector<CTrajectory> >::iterator validTra_map_it,invalidTra_map_it;
        map<ulong,_trajectory> trajectory_team;
        map<ulong,_trajectory>::iterator tra_team_it;

        for (validTra_map_it = validTra_map.begin();validTra_map_it != validTra_map.end();validTra_map_it++)
        {
            ulong startFrame = validTra_map_it->first;
            invalidTra_map_it = invalidTra_map.find(startFrame);
            _trajectory tmp;
            if (invalidTra_map_it != invalidTra_map.end())
            {               
                tmp.m_validTrajectory = validTra_map_it->second;
                tmp.m_invalidTrajectory = invalidTra_map_it->second;
                trajectory_team.insert(std::pair<ulong,_trajectory>(startFrame,tmp));
            }
            else
            {
                tmp.m_validTrajectory = validTra_map_it->second;
                tmp.m_validTrajectory = vector<CTrajectory>();
                trajectory_team.insert(std::pair<ulong,_trajectory>(startFrame,tmp));
            }
        }

        trajectory_team_cluseter.push_back(trajectory_team);
    }

#ifdef _DEBUG
    // 验证输出是否正确。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    // 将同一个批次的点依次打印在相应的帧上面。
    vector<CTrajectory> outputVerify;  
    vector<CTrajectory> invalidFeaturesVerify;
#endif
    

    ofstream outfile(".\\info.txt");
    for (size_t i = 0;i < trajectory_team_cluseter.size();i++)
    {
        outfile<<"cluster:"<<i + 1<<endl;
        map<ulong,_trajectory>& trajectory_team = trajectory_team_cluseter[i];
        map<ulong,_trajectory>::iterator tra_team_it;
        int index = 1;
        for (tra_team_it = trajectory_team.begin();tra_team_it != trajectory_team.end();tra_team_it++)
        {
            outfile<<"team:"<<index++;
            vector<CTrajectory>& valid_tra_vec = tra_team_it->second.m_validTrajectory;
            vector<CTrajectory>& invalid_tra_vec = tra_team_it->second.m_invalidTrajectory;
            outfile<<" valid features:"<<valid_tra_vec.size()<<endl;

#ifdef _DEBUG
            if (i == 0 && index == 15)               
            {
                outputVerify = valid_tra_vec;
                cout<<"outputVerify size:"<<outputVerify.size()<<endl;
                invalidFeaturesVerify = invalid_tra_vec;
                cout<<"invalidFeaturesVerify size:"<<invalidFeaturesVerify.size()<<endl;
            }
#endif

            // 获取同一批次轨迹中包含的最长的帧数maxFrame，则同一批次轨迹共需输出maxFrame行。
            size_t maxFrame = 0;
            for (size_t j = 0;j < valid_tra_vec.size();j++)
            {
                vector<Point2f>& trajectory = valid_tra_vec[j].getTrajectory();
                size_t nframe = trajectory.size();
                if (nframe > maxFrame) maxFrame = nframe;
            }
            for (size_t j = 0;j < invalid_tra_vec.size();j++)
            {
                vector<Point2f>& trajectory = invalid_tra_vec[j].getTrajectory();
                size_t nframe = trajectory.size();
#ifdef _DEBUG
                if (i == 0 && index == 15)
                {
                    if (nframe > 2)
                        cout<<"size is more than 2"<<endl;
                }
#endif
                if (nframe > maxFrame) maxFrame = nframe;
            }

            // 输出同一批次的估计在各个帧中的位置信息。
            for (size_t j = 0;j < maxFrame;j++)
            {
                // 有效轨迹
                for (size_t k = 0;k < valid_tra_vec.size();k++)
                {
                    vector<Point2f>& trajectory = valid_tra_vec[k].getTrajectory();
                    size_t size = trajectory.size();
                    if (j >= size)
                        outfile<<trajectory[size - 1];
                    else
                        outfile<<trajectory[j];
                    outfile<<" ";
                }
                // 无效轨迹
                for (size_t k = 0;k < invalid_tra_vec.size();k++)
                {
                    vector<Point2f>& trajectory = invalid_tra_vec[k].getTrajectory();
                    size_t size = trajectory.size();
                    if (j >= size)
                        outfile<<trajectory[size - 1];
                    else
                        outfile<<trajectory[j];
                    outfile<<" ";
                }
                outfile<<endl;
            }
        }
    }

    /*
    ofstream outfile(".\\info.txt");
    for (size_t i = 0;i < clusTra_vec.size();i++)
    {
        outfile<<"cluster:"<<i + 1<<endl;
        map<ulong,vector<CTrajectory> >& tra_map = clusTra_vec[i];
        map<ulong,vector<CTrajectory> >::iterator it;
        int index = 1;
        for (it = tra_map.begin();it != tra_map.end();it++)
        {
            outfile<<"team:"<<index++<<endl;
            vector<CTrajectory>& tra_vec = it->second;

#ifdef _DEBUG
            if (i == 0 && index == 15)
                outputVerify = tra_vec;
#endif

            // 获取同一批次轨迹中包含的最长的帧数maxFrame，则同一批次轨迹共需输出maxFrame行。
            size_t maxFrame = 0;
            for (size_t j = 0;j < tra_vec.size();j++)
            {
                vector<Point2f>& trajectory = tra_vec[j].getTrajectory();
                size_t nframe = trajectory.size();
                if (nframe > maxFrame) maxFrame = nframe;
            }

            // 输出同一批次的估计在各个帧中的位置信息。
            for (size_t j = 0;j < maxFrame;j++)
            {
                for (size_t k = 0;k < tra_vec.size();k++)
                {
                    vector<Point2f>& trajectory = tra_vec[k].getTrajectory();
                    size_t size = trajectory.size();
                    if (j >= size)
                        outfile<<trajectory[size - 1];
                    else
                        outfile<<trajectory[j];
                    outfile<<" ";
                }
                outfile<<endl;
            }
        }
    }
    */
    

#ifdef _DEBUG

    VideoCapture video(videopath);
    if (!video.isOpened())
    {
        cerr<<"can not open the video:"<<videopath<<endl;
        return;
    }
    Mat Frame;
    for (size_t i = 0;i < 10;i++) video >> Frame;   // 去掉开始的10帧图片

    ulong nFrame = 0,index = 0;
    char szverify[256];
    memset(szverify,0,sizeof(szverify));
    ulong startFrame,endFrame = 0;
    // 所有轨迹具有相同的起始帧
    startFrame = outputVerify[0].getStartFrame();
    // 选取最后结束的帧
    for (size_t i = 0;i < outputVerify.size();i++)
    {
        ulong tmp = outputVerify[i].getEndFrame();
        if (tmp > endFrame) endFrame = tmp;
    }
    for (size_t i = 0;i < invalidFeaturesVerify.size();i++)
    {
        ulong tmp = invalidFeaturesVerify[i].getEndFrame();
        if (tmp > endFrame) endFrame = tmp;
    }

    for (ulong i = 0;i < startFrame;i++) video >> Frame;

    ofstream verifyfile(".\\verify.txt");
    for (ulong i = startFrame;i <= endFrame;i++)
    {
        video >> Frame;
/*
        for (size_t j = 0;j < outputVerify.size();j++)
        {
            vector<Point2f>& trajectory = outputVerify[j].getTrajectory();
            size_t size = trajectory.size();           
            int red,blue,green;
            if (j % 4 == 0)
            {
                red = 255;
                blue = green = 0;
            }
            else if (j % 4 == 1)
            {
                green = 255;
                red = blue = 0;
            }
            else if (j % 4 == 2)
            {
                blue = 255;
                red = green = 0;
            }
            else
            {
                red = green = 255;
                blue = 0;
            }
            if (index >= size)
                circle(Frame,trajectory[size - 1],2,CV_RGB(red,green,blue),-1);
            else
                circle(Frame,trajectory[index],2,CV_RGB(red,green,blue),-1);
        }
*/
        for (size_t j = 0;j < outputVerify.size();j++)
        {
            vector<Point2f>& trajectory = outputVerify[j].getTrajectory();
            size_t size = trajectory.size();  
            if (index >= size)
                circle(Frame,trajectory[size - 1],2,CV_RGB(255,255,0),-1);
            else
                circle(Frame,trajectory[index],2,CV_RGB(255,255,0),-1);
        }
        for (size_t j = 0;j < invalidFeaturesVerify.size();j++)
        {
            vector<Point2f>& trajectory = invalidFeaturesVerify[j].getTrajectory();
            size_t size = trajectory.size();  
            if (size > 2)
            {
                cout<<"invalid features size is more than 2"<<endl;
            }
            if (index >= size)
            {
                circle(Frame,trajectory[size - 1],2,CV_RGB(0,0,255),-1);
                verifyfile<<trajectory[size - 1];
            }
            else
            {
                circle(Frame,trajectory[index],2,CV_RGB(0,0,255),-1);
                verifyfile<<trajectory[index];
            }
            verifyfile<<" ";
        }
        verifyfile<<endl;
        rectangle(Frame,Rect(45,106,230,28),CV_RGB(0,0,0),1);
        rectangle(Frame,Rect(45,134,230,10),CV_RGB(0,0,0),1);
        sprintf_s(szverify,".\\verify\\%d.jpg",index++);
        imwrite(szverify,Frame);
    }

#endif

}

void GetTripwireRectAndDetectRect(cv::Size& videoSize,cv::Size& detectSize,int tripwireHeight,cv::Rect& tripwireRect,cv::Rect& detectRect)
{
    // 检测窗口处于图像的正中位置
    int topLeft_x = (videoSize.width - detectSize.width) >> 1;
    int topLeft_y = (videoSize.height - detectSize.height) >> 1;
    int width = detectSize.width;
    int height = detectSize.height;

    detectRect.x = topLeft_x;
    detectRect.y = topLeft_y;
    detectRect.width = width;
    detectRect.height = height;

    // 特征点检测窗口位于检测窗口图像下方，紧接检测窗口
    // 特征点检测窗口的宽度与检测窗口的宽度一致
    topLeft_y += detectSize.height;
    height = tripwireHeight;

    tripwireRect.x = topLeft_x;
    tripwireRect.y = topLeft_y;
    tripwireRect.width = width;
    tripwireRect.height = height;
}

void start_end_variance(vector<Point>& points,float& startVar,float& endVar,int time_per_frame)
{
    startVar = endVar = 0.f;
    float startMean = 0.f,endMean = 0.f;
    vector<Point>::size_type size = points.size();

    for (vector<Point>::size_type i = 0;i < size;i++)
    {
        startMean += points[i].x;
        endMean += points[i].y;
        startVar += points[i].x * points[i].x;
        endVar += points[i].y * points[i].y;
    }

    startMean /= size;
    endMean /= size;
    startVar = sqrt(startVar / size - startMean * startMean) * time_per_frame;    // 转换成实际的时间
    endVar = sqrt(endVar / size - endMean * endMean) * time_per_frame;
}

void lifetime_mean_variance(vector<Point>& points,float& lifetimeMean,float& lifetimeVar,int time_per_frame)
{
    lifetimeMean = lifetimeVar = 0.f;
    vector<Point>::size_type size = points.size();

    for (vector<Point>::size_type i = 0;i < size;i++)
    {
        lifetimeMean += (points[i].y - points[i].x);
        lifetimeVar += (points[i].y - points[i].x) * (points[i].y - points[i].x);
    }

    lifetimeMean = lifetimeMean / size;
    lifetimeVar = sqrt(lifetimeVar / size - lifetimeMean * lifetimeMean) * time_per_frame;
    lifetimeMean = lifetimeMean * time_per_frame;
}