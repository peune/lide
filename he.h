
#ifndef _HE_H_
#define _HE_H_

#include <opencv2/opencv.hpp>

float compute_hist( cv::Mat& img, float* hist );

void equalize_hist( int nlevel, float *hist, int *newval );

void equalize_hist_interval( int L, float* hist, int M, int* thresholds, int* newval );

float grayscale_remap( cv::Mat& img, int* newval, cv::Mat& out );

 
void HE( cv::Mat& img, cv::Mat& out );

#endif // _HE_H_

