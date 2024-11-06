
#ifndef _LIDE_SIMPLE_H_
#define _LIDE_SIMPLE_H_

#include <opencv2/opencv.hpp>

void LIDE_simple( cv::Mat& img, cv::Mat& out, int model, int d, float sigma_min );

void LIDEHM_simple( cv::Mat& img, cv::Mat& out, int model, int d, float sigma_min, int n, float* hist );

void LIDE_simple( cv::Mat& img, cv::Mat& out, int model, int d, float sigma_min, int target_distrib );



void compute_local_stat( cv::Mat& img, int d, cv::Mat& mean, cv::Mat& sd, float sigma_min );

void lides_reset_mem_counter();

#endif // _LIDE_SIMPLE_H_


