#include "stdafx.h"
#include "videoParams.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
using namespace una_videoparams;


int main(int argc,char* argv[])
{
    
    CVideoParams videoparam;
    videoparam.Initialize(".\\para.json");
    //videoparam.featuresDistribute("G:\\work\\videos\\FD_441_12_00.avi",".\\testresult\\featureDistribute\\FD_441_12_00");
    //videoparam.validFeaturePoints("G:\\work\\videos\\FD_441_12_00.avi",".\\testresult\\validFeatures\\FD_441_12_00");
    //videoparam.trackFailurePoints("F:\\multi\\FD_001_06_00.avi",".\\testresult\\trackFailure\\FD_001_06_00");
	//videoparam.invalidTrajectorys("G:\\work\\videos\\FD_001_06_00.avi",".\\testresult\\invalidTrajectorys\\FD_001_06_00");
    videoparam.featuresStatus("F:\\multi\\FD_001_06_00.avi",".\\testresult\\trackFailure\\FD_001_06_00");
    return 1;
}

