

#ifndef _AHE_H_
#define _AHE_H_

#include <opencv2/opencv.hpp>


void AHE( cv::Mat& img, cv::Mat& out, int d, float clipping=0.01, bool rescale_intensity=false );

void AHEHM( cv::Mat& img, cv::Mat& out, int d, float clipping, float* target );


struct FastLocalHist
{
	int nlevel;
	int w, h;
	int*** ii;

	FastLocalHist( int new_w, int new_h, int new_nlevel=256 );
	~FastLocalHist();
	
	void init( cv::Mat& img, int x0, int y0 );
	void compute_hist( int x0, int y0, int x1, int y1, float clipping, float* hist );
};



#endif // _AHE_H_
