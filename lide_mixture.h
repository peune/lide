
#ifndef _LIDE_MIXTURE_H_
#define _LIDE_MIXTURE_H_

#include <opencv2/opencv.hpp>

void LIDE_mixture( cv::Mat& org, cv::Mat& out, int model, int K, int d, float sigma_min );

void LIDEHM_mixture( cv::Mat& org, cv::Mat& out, int model, int K, int d, float sigma_min, int n, float* hist );

void LIDE_mixture( cv::Mat& img, cv::Mat& out, int model, int K, int d, float sigma_min, int target_distrib );



void lidem_reset_mem_counter();

#endif // _LIDE_MIXTURE_H_
