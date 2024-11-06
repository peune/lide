
#ifndef _PS_H_
#define _PS_H_

#include <opencv2/opencv.hpp>


float compute_local_mean( cv::Mat& img, int d, cv::Mat& mean );

/*
  phi: ...

  target
  0 : HE
  1 : PHEME
  2 : FSHABP
 */
void PS( cv::Mat& img, cv::Mat& out, int d, int phi, int target );

void PSreg( cv::Mat& img, cv::Mat& out, int d, int phi, int target_distrib, 
	    float lambda );

void PScolor( cv::Mat& img, cv::Mat& rgb_img, cv::Mat& out, int d, int phi, int target_distrib );

#endif // _PS_H_

