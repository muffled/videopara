#include "stdafx.h"
#include "Validation.h"

bool CTrajectoryValidation::validation(CTrajectory& trajectory)
{
    // �˴���Ҫ��֡��תΪ��Ӧ��ʱ��
    ulong lifetime = trajectory.getLifeTime() * m_timePerFrame;
    if ( abs(lifetime - m_meanLifeTime) >= 2.5*m_varLifeTime ) return false;
    return true;
}