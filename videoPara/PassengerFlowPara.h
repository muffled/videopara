#pragma once

#include "json/json.h"
#include "vani_log.h"
#include <fstream>
#include <iostream>

typedef unsigned long ulong;

// ����������ز���
struct InitPara
{
    // ѵ����Ƶ��ͼƬ�ߴ�
    ulong m_video_width;
    ulong m_video_height;

    // ������׷�ٴ���
    ulong m_detect_width;
    ulong m_detect_height;

    // �������ⴰ��
    ulong m_tripwire_height;

    // ������֮֡���ʱ��(���룩
    ulong m_time_per_frame;

    // �켣��֤����
    ulong m_life_mean;
    ulong m_life_std;

    // �켣�������
    ulong m_ts_std;
    ulong m_te_std;
    ulong m_sampleThresh;
};

// ������������
struct PassengerFlowPara
{
    // �켣�ľ���
    int m_start_std;                 // �켣��ʼʱ��ķ���
    int m_end_std;                   // �켣����ʱ��ķ���

    // �켣����֤
    int m_life_mean;              // �켣�����ڵľ�ֵ
    int m_life_std;               // �켣�����ڵ�����

    // ��־�˵������˵Ĺ켣����
    int m_trajectory_quantity;       
};

// ��ָ�����ļ��ж�ȡ��ز�����ʧ�ܷ���false.
//bool readInitPara(const char* szfile,struct InitPara& para);

// �������ͳ���������
//void writePassengerFlowPara(const char* szfile,struct PassengerFlowPara& para);
