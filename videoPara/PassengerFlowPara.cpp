#include "PassengerFlowPara.h"

using namespace std;


// bool readInitPara(const char* szfile,struct InitPara& para)
// {
//     ifstream file(szfile);
//     if (!file.is_open())
//     {
//         string slog("be unable to open the specified file:");
//         slog += szfile;
//         LogWriter(slog.c_str());
//         return false;
//     }
// 
//     Json::Reader reader;
//     Json::Value root;
//     if (!reader.parse(file,root,false))
//     {
//         string slog("json parse the specified file:");
//         slog = slog + szfile +" failed.";
//         LogWriter(slog.c_str());
//         return false;
//     }
// 
//     para.m_video_width = root.get("video_width",320).asInt();
//     para.m_video_height = root.get("video_height",240).asInt();
// 
//     para.m_detect_width = root.get("detect_width",230).asInt();
//     para.m_detect_height = root.get("detect_height",28).asInt();
// 
//     para.m_tripwire_height = root.get("tripwire_height",10).asInt();
// 
//     para.m_time_per_frame = root.get("time_per_frame",40).asInt();
// 
//     para.m_life_mean = root.get("life_mean",251).asInt();
//     para.m_life_std = root.get("life_std",150).asInt();
// 
//     para.m_ts_std = root.get("ts_std",364).asInt();
//     para.m_te_std = root.get("te_std",359).asInt();
//     para.m_sampleThresh = root.get("sampleThresh",10).asInt();
// 
//     return true;
// }

// void writePassengerFlowPara(const char* szfile,struct PassengerFlowPara& para)
// {
//     Json::Value root;
//     root["start_std"] = para.m_start_std;
//     root["end_std"] = para.m_end_std;
//     root["life_mean"] = para.m_life_mean;
//     root["life_std"] = para.m_life_std;
//     root["trajectory_quantity"] = para.m_trajectory_quantity;
// 
//     Json::StyledWriter writer;
//     std::string sres = writer.write(root);
// 
//     ofstream outfile(szfile);
//     outfile<<sres;
// }