#pragma once

#include "stdafx.h"
#include "Trajectory.h"
#include "FeatureTrack.h"
#include "Validation.h"
#include "cluster.h"
#include "PassengerFlowPara.h"
#include <map>

class CTrajectory;
struct clusterPoints;

namespace una_videoparams
{
    struct traAndColor
    {
        int m_red;
        int m_green;
        int m_blue;
        std::vector<cv::Point2f> m_points;
    };

	struct colorPoint
	{
		int m_red;
		int m_green;
		int m_blue;
		cv::Point2f m_point;
	};

	// 记录下每个聚类的信息。
	struct clusterInfo
	{
		int startFrame;  // 聚类最初开始的帧数（聚类所有轨迹中最早出现的帧数）
		int endFrame;    // 聚类最终结束的帧数（聚类所有轨迹中最后结束的帧数）
		int totalTrajectorys; // 聚类所包含的轨迹的数量
		float aveSpeed;  // 聚类的平均速度
	};

    class CVideoParams
    {                                        
    private:
        // m_featurePerFrame 用于记录下每一帧中所有特征点的位置（如果存在）
        // 在视频第二次播放的时候，可以在每一帧中将特征点在视频中打印出来。
        std::map<int,std::vector<cv::Point2f> > m_featuresPerFrame;
		std::map<int,std::vector<colorPoint> > m_featuresPerFrame_color;
        std::string m_videoPath;          // 视频文件的路径

        struct InitPara m_initParam;
        struct PassengerFlowPara m_passengerFlowParam;

        std::list<CTrajectory> m_listTrajectory;
        std::list<CTrajectory> m_list_finishedTrajectory;   // 存储生命期结束的轨迹，用于后续的认证和聚类
        std::list<CTrajectory> m_list_unfinishedTrajectory;
     
        CLKTTracker m_tracker;                // 特征点的追踪
        CTrajectoryValidation m_validation;  // 轨迹的认证
        CTrajectoryCluster m_cluster;        // 轨迹的聚类
    protected:
        bool readInitPara(std::string spath);
        void writePassengerFlowPara(std::string outpath);
        void calcParam(std::vector<clusterPoints>&);
        void getFeaturesPerFrame(std::vector<clusterTrajectory>&);
        void getFeaturesPerFrame(std::vector<clusterTrajectory>&,int);
        void playVideo(std::string videopath);
    public:
        CVideoParams() {};
        bool Initialize(std::string spath);
        void videoProcess(std::string videopath,std::string resultpath);

        /**
        对于每个人，统计出检测到的初始特征点的数目及最终成为有效特征点的数目。
        
        第一次先处理完视频，得到所有有效的轨迹。第二次播放视频时候，每个有效的轨迹的初始点所在的帧
        即为检测到初始特征点所在的帧，保存下此帧的图片。同时打印出此帧检测窗口中所有检测到的特征点，
        如果某一特征点不在有效轨迹中，则表明此特征点不是有效的特征点。将有效特征点与无效特征点以不
        同的颜色打印。

        @param videopath 待处理的视频的路径
        @param resultpath 结果图片保存文件夹。
        **/
        void validFeaturePoints(std::string videopath,std::string resultpath);
        
        /**
        对于一段视频，显示出每个不能有效追踪的特征点。
        实现方法：
        第一次处理完视频，得到所有由于跟踪失败而遭到抛弃的轨迹，所有这些轨迹的最后一个点即为
        不能正确跟踪的特征点。
        第二次播放视频的时候，打印出所有的这些特征点。

        @param videopath 待处理的视频的路径
        @param resultpath 结果图片保存文件夹。
        **/
        void trackFailurePoints(std::string videopath,std::string resultpath); 

		/**
		对于一段视频，显示出整个聚类的特征点分布。
		实现方法：
		第一次处理完视频，根据所得到的聚类结果，得到每一帧图片中所包含的所有有效轨迹在此帧中对应的特征点。
		以帧数为键值，所有特征点的vector为元素，存储到字典中。
		第二次播放时，根据字典中相应的键值，打印出此帧所对应的所有的特征点。

		@param videopath 待处理的视频的路径
		@param resultpath 结果图片保存文件夹。
		**/
		void featuresDistribute(std::string videopath,std::string resultpath);

		void invalidTrajectorys(std::string videopath,std::string resultpath);

        /**
        对于每一个聚类，打印出每一条轨迹在每帧中的位置。
        聚类中，将开始时间相同的轨迹（轨迹起始点处于相同的帧数）作为一组，将特征点的位置按组打印出来。

        格式：
        cluster 1:
        team 1:
        point matrix
        team 2:
        point matrix
        .......
        cluster 2:
        .......

        point matrix 的行数代表了轨迹占据的帧数，列数代表了此组中轨迹的数量。
        **/
        void featuresStatus(std::string videopath,std::string resultpath);
    };
}