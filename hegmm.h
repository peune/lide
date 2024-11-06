
#ifndef _HEGMM_H_
#define _HEGMM_H_

/*
  HEGMM
  T. Celik and T. Tjahjadi, "Automatic Image Equalization and Contrast Enhancement Using Gaussian Mixture Modeling", 
  IEEE Transactions on Image Processing 21, 1 (2012), pp. 145-156.

  Idea:
  - Train GMM 
    This paper uses Figuerido-Jain EM style training. This procedure is based on the idea of Component-wise EM Algorithm for Mixtures or CEM
    of Celeux et al. CEM iteratively applies EM by updating one component each time. 

 - Partition the grayscale range into intervals by analyzing the intersection between Gaussian components
 - For each interval, we identifies the dominant Gaussian component. 
 - Each interval is weighted by mixing the CDF of the dominant Gaussian with its variance.
 - The Gaussian CDF is used as the mapping function in each interval.

 - The pixels are grouped by their grayscale value to speed up the computation
 */

void HEGMM( cv::Mat& org, cv::Mat& out, int K );

void init_gmm( cv::Mat& img, int K, float* mu, float* sigma, float* pi, float& xmin, float& xmax );

void find_significant_intersection( int K, float* mu, float* sigma, float* pi,
				    int xmin, int xmax, 
				    int& l, float* xs ) ;

void find_dominant_gaussian( int K, float* mu, float* sigma, float* pi, 
			     int l, float* xs,
			     int* dominant_gauss );


#endif // _HEGMM_H_
