#include "stdafx.h"
#include "Validation.h"

bool CTrajectoryValidation::validation(CTrajectory& trajectory)
{
    // 此处需要将帧数转为相应的时间
    ulong lifetime = trajectory.getLifeTime() * m_timePerFrame;
    if ( abs(lifetime - m_meanLifeTime) >= 2.5*m_varLifeTime ) return false;
    return true;
}