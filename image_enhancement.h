
#ifndef _IMAGE_ENHANCEMENT_H_
#define _IMAGE_ENHANCEMENT_H_

#define ENHANCE_HE       0
#define ENHANCE_AHE      1
#define ENHANCE_DHE      2
#define ENHANCE_MHE      3
#define ENHANCE_ESIHE    4
#define ENHANCE_BPHEME   5
#define ENHANCE_FHSABP   6
#define ENHANCE_HEGMM    7
#define ENHANCE_LIDEG    8
#define ENHANCE_LIDEL    9
#define ENHANCE_LIDEGMM  10
#define ENHANCE_LIDELMM  11

#include <opencv2/opencv.hpp>

struct EnhanceParam
{
	int method;
	
	int d;       // size of local window is (2d+1)x(2d+1), default = 100
	int target;  // output target, 0~HE, 1~BPHEME, 2~FSHABP

	float ahe_clipping;          // perform relative discounting -> clip all histogram proportionally and redistribute the count, default = 0.01
	bool  ahe_rescale_intensity; // rescale intensity such that the output mean matches that of the input, default = false

	int dhe_min_length;   // minimum length of each sub-interval, default=20

	int mhe_mmin;         // minimum number of intervals, default = 5
	int mhe_mmax;         // maximum number of intervals, default = 10
	float mhe_rho;        // regularization coeff for finding optimal number of intervals, default = 0.8

	int hegmm_k;          // number of components, default = 20

	int lide_mod;         // proba model, 0=Gaussian, 1=Laplacian
	int lide_k;           // number of components, default = 10
	float lide_sigma_min; // minimum value for sigma to avoid instability in uniform area, default = 50; 

	EnhanceParam();
};

void enhance( cv::Mat& org, cv::Mat& out, EnhanceParam &param );

void color_enhance( cv::Mat& rgb_org, cv::Mat& rgb_out, EnhanceParam &param );

#endif // _IMAGE_ENHANCEMENT_H_




