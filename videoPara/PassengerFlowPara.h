#pragma once

#include "json/json.h"
#include "vani_log.h"
#include <fstream>
#include <iostream>

typedef unsigned long ulong;

// 程序运行相关参数
struct InitPara
{
    // 训练视频的图片尺寸
    ulong m_video_width;
    ulong m_video_height;

    // 特征点追踪窗口
    ulong m_detect_width;
    ulong m_detect_height;

    // 特征点检测窗口
    ulong m_tripwire_height;

    // 相邻两帧之间的时间(毫秒）
    ulong m_time_per_frame;

    // 轨迹验证参数
    ulong m_life_mean;
    ulong m_life_std;

    // 轨迹聚类参数
    ulong m_ts_std;
    ulong m_te_std;
    ulong m_sampleThresh;
};

// 程序的输出参数
struct PassengerFlowPara
{
    // 轨迹的聚类
    int m_start_std;                 // 轨迹开始时间的方差
    int m_end_std;                   // 轨迹结束时间的方差

    // 轨迹的验证
    int m_life_mean;              // 轨迹生命期的均值
    int m_life_std;               // 轨迹生命期的数量

    // 标志此单个行人的轨迹数量
    int m_trajectory_quantity;       
};

// 从指定的文件中读取相关参数，失败返回false.
//bool readInitPara(const char* szfile,struct InitPara& para);

// 输出客流统计所需参数
//void writePassengerFlowPara(const char* szfile,struct PassengerFlowPara& para);
