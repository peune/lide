

#include "he.h"
#include "ahe.h"
#include "dhe.h"
#include "mhe.h"
#include "bpheme.h"
#include "fhsabp.h"
#include "lide_simple.h"
#include "lide_mixture.h"
#include "hegmm.h"
#include "esihe.h"

#include "image_enhancement.h"

#include <opencv2/opencv.hpp>
using namespace cv;

EnhanceParam::EnhanceParam()
{
	method = ENHANCE_HE;;
	
	d = 100;
	target = 0;

	ahe_clipping          = 0.05;
	ahe_rescale_intensity = false;

	dhe_min_length = 20;

	mhe_mmin = 5;
	mhe_mmax = 10;
	mhe_rho  = 0.8;

	hegmm_k  = 20;

	lide_mod = 0;
	lide_k   = 10;
	lide_sigma_min = 30;
}


void enhance( cv::Mat& org, cv::Mat& out, EnhanceParam &param )
{
	switch(param.method)
	{
	case ENHANCE_HE      : HE( org, out );                                                 break;
	case ENHANCE_AHE     : AHE( org, out, param.d, param.ahe_clipping, param.ahe_rescale_intensity ); break; 
	case ENHANCE_DHE     : DHE( org, out, param.dhe_min_length );                          break; 
	case ENHANCE_MHE     : MHE( org, out, param.mhe_mmin, param.mhe_mmax, param.mhe_rho ); break; 
	case ENHANCE_ESIHE   : ESIHE( org, out );  break;
	case ENHANCE_BPHEME  : BPHEME( org, out ); break;
	case ENHANCE_FHSABP  : FHSABP( org, out ); break;
	case ENHANCE_HEGMM   : HEGMM( org, out, param.hegmm_k );     break;
	case ENHANCE_LIDEG   : LIDE_simple( org, out, 0, param.d, param.lide_sigma_min, param.target );  break;
	case ENHANCE_LIDEL   : LIDE_simple( org, out, 1, param.d, param.lide_sigma_min, param.target );  break;
	case ENHANCE_LIDEGMM : LIDE_mixture( org, out, 0, param.lide_k, param.d, param.lide_sigma_min, param.target );  break;
	case ENHANCE_LIDELMM : LIDE_mixture( org, out, 1, param.lide_k, param.d, param.lide_sigma_min, param.target );  break;
	}
}


void color_enhance( cv::Mat& rgb_org, cv::Mat& rgb_out, EnhanceParam &param )
{
	int w = rgb_org.cols, h = rgb_org.rows;
	Mat org( h, w, CV_8UC1 );
	Mat out( h, w, CV_8UC1 );

	for( int x = 0; x < w; x++ )
	{
		for( int y = 0; y < h; y++ )
		{
			int g = (rgb_org.at<Vec3b>(y,x)[0] + 
				 rgb_org.at<Vec3b>(y,x)[1] +
				 rgb_org.at<Vec3b>(y,x)[2]) / 3;

			org.at<uchar>(y,x) = g;
		}
	}

	enhance( org, out, param );

	for( int x = 0; x < w; x++ )
	{
		for( int y = 0; y < h; y++ )
		{
			float f = (float)out.at<uchar>(y,x) / org.at<uchar>(y,x);

			rgb_out.at<Vec3b>(y,x)[0] = saturate_cast<uchar> (rgb_org.at<Vec3b>(y,x)[0] * f);
			rgb_out.at<Vec3b>(y,x)[1] = saturate_cast<uchar> (rgb_org.at<Vec3b>(y,x)[1] * f);
			rgb_out.at<Vec3b>(y,x)[2] = saturate_cast<uchar> (rgb_org.at<Vec3b>(y,x)[2] * f);
		}
	}
}



