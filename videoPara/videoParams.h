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

	// ��¼��ÿ���������Ϣ��
	struct clusterInfo
	{
		int startFrame;  // ���������ʼ��֡�����������й켣��������ֵ�֡����
		int endFrame;    // �������ս�����֡�����������й켣����������֡����
		int totalTrajectorys; // �����������Ĺ켣������
		float aveSpeed;  // �����ƽ���ٶ�
	};

    class CVideoParams
    {                                        
    private:
        // m_featurePerFrame ���ڼ�¼��ÿһ֡�������������λ�ã�������ڣ�
        // ����Ƶ�ڶ��β��ŵ�ʱ�򣬿�����ÿһ֡�н�����������Ƶ�д�ӡ������
        std::map<int,std::vector<cv::Point2f> > m_featuresPerFrame;
		std::map<int,std::vector<colorPoint> > m_featuresPerFrame_color;
        std::string m_videoPath;          // ��Ƶ�ļ���·��

        struct InitPara m_initParam;
        struct PassengerFlowPara m_passengerFlowParam;

        std::list<CTrajectory> m_listTrajectory;
        std::list<CTrajectory> m_list_finishedTrajectory;   // �洢�����ڽ����Ĺ켣�����ں�������֤�;���
        std::list<CTrajectory> m_list_unfinishedTrajectory;
     
        CLKTTracker m_tracker;                // �������׷��
        CTrajectoryValidation m_validation;  // �켣����֤
        CTrajectoryCluster m_cluster;        // �켣�ľ���
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
        ����ÿ���ˣ�ͳ�Ƴ���⵽�ĳ�ʼ���������Ŀ�����ճ�Ϊ��Ч���������Ŀ��
        
        ��һ���ȴ�������Ƶ���õ�������Ч�Ĺ켣���ڶ��β�����Ƶʱ��ÿ����Ч�Ĺ켣�ĳ�ʼ�����ڵ�֡
        ��Ϊ��⵽��ʼ���������ڵ�֡�������´�֡��ͼƬ��ͬʱ��ӡ����֡��ⴰ�������м�⵽�������㣬
        ���ĳһ�����㲻����Ч�켣�У�������������㲻����Ч�������㡣����Ч����������Ч�������Բ�
        ͬ����ɫ��ӡ��

        @param videopath ���������Ƶ��·��
        @param resultpath ���ͼƬ�����ļ��С�
        **/
        void validFeaturePoints(std::string videopath,std::string resultpath);
        
        /**
        ����һ����Ƶ����ʾ��ÿ��������Ч׷�ٵ������㡣
        ʵ�ַ�����
        ��һ�δ�������Ƶ���õ��������ڸ���ʧ�ܶ��⵽�����Ĺ켣��������Щ�켣�����һ���㼴Ϊ
        ������ȷ���ٵ������㡣
        �ڶ��β�����Ƶ��ʱ�򣬴�ӡ�����е���Щ�����㡣

        @param videopath ���������Ƶ��·��
        @param resultpath ���ͼƬ�����ļ��С�
        **/
        void trackFailurePoints(std::string videopath,std::string resultpath); 

		/**
		����һ����Ƶ����ʾ�����������������ֲ���
		ʵ�ַ�����
		��һ�δ�������Ƶ���������õ��ľ��������õ�ÿһ֡ͼƬ����������������Ч�켣�ڴ�֡�ж�Ӧ�������㡣
		��֡��Ϊ��ֵ�������������vectorΪԪ�أ��洢���ֵ��С�
		�ڶ��β���ʱ�������ֵ�����Ӧ�ļ�ֵ����ӡ����֡����Ӧ�����е������㡣

		@param videopath ���������Ƶ��·��
		@param resultpath ���ͼƬ�����ļ��С�
		**/
		void featuresDistribute(std::string videopath,std::string resultpath);

		void invalidTrajectorys(std::string videopath,std::string resultpath);

        /**
        ����ÿһ�����࣬��ӡ��ÿһ���켣��ÿ֡�е�λ�á�
        �����У�����ʼʱ����ͬ�Ĺ켣���켣��ʼ�㴦����ͬ��֡������Ϊһ�飬���������λ�ð����ӡ������

        ��ʽ��
        cluster 1:
        team 1:
        point matrix
        team 2:
        point matrix
        .......
        cluster 2:
        .......

        point matrix �����������˹켣ռ�ݵ�֡�������������˴����й켣��������
        **/
        void featuresStatus(std::string videopath,std::string resultpath);
    };
}