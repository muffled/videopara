#pragma once

#include "stdafx.h"
#include "Trajectory.h"

class CTrajectoryValidation
{
private:
    double m_meanLifeTime;      
    double m_varLifeTime;
    int    m_timePerFrame;
public:
    CTrajectoryValidation():m_meanLifeTime(0),m_varLifeTime(0) {}
    CTrajectoryValidation(double mean,double var):m_meanLifeTime(mean),m_varLifeTime(var) {}

    inline double getMeanLifeTime()  {return m_meanLifeTime;}
    inline double getVarLifeTime()   {return m_varLifeTime;}

    inline void setMeanLifeTime(double mean) {m_meanLifeTime = mean;}
    inline void setVarLifeTime(double var)   {m_varLifeTime = var;}
    inline void setTimePerFrame(int timePerFrame)  {m_timePerFrame = timePerFrame;}

    bool validation(CTrajectory&);
};