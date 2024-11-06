
#ifndef _MHE_H_
#define _MHE_H_

#include <opencv2/opencv.hpp>

void MHE( cv::Mat& img, cv::Mat& out, int Mmin, int Mmax, float rho );


int mhe_partition( int L, float* hist, int* thresholds, int Mmin, int Mmax, float rho );

#endif // _MHE_H_


