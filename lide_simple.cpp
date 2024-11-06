
#include <iostream>
using namespace std;

#include "lide_simple.h"
using namespace cv;

double _lides_mem_counter;

void lides_reset_mem_counter() { _lides_mem_counter = 0; }


///////////////////////////////////////////////////////////////////////////
void compute_local_stat( Mat& img, int d, Mat& mean, Mat& sd, float sigma_min ) // sigma_min = 50
{
	// allocate tmp vars
	int w = img.cols;
	int h = img.rows;
	float **ii  = new float* [w];
	float **ii2 = new float* [w];
	float **s   = new float* [w];
	float **s2  = new float* [w];
	for( int x = 0; x < w; x++ ) 
	{
		ii[x]  = new float [h];
		ii2[x] = new float [h];
		s[x]   = new float [h];
		s2[x]  = new float [h];
	}

	double  ff = (w * h)/1048576.0;
	_lides_mem_counter += (4.0 * ff);
	cout<< "LIDES MEM>> "<< _lides_mem_counter<< endl;
	

	// compute integral images
	for( int x = 0; x < w; x++ ) 
	{
		for( int y = 0; y < h; y++ ) 
		{
			float p = (float)img.at<unsigned char>(y,x) / 255.0;

			s[x][y]  = (y==0 ? 0 :  s[x][y-1]) + p;
			ii[x][y] = (x==0 ? 0 : ii[x-1][y]) + s[x][y];

			s2[x][y]  = (y==0 ? 0 :  s2[x][y-1]) + p*p;
			ii2[x][y] = (x==0 ? 0 : ii2[x-1][y]) + s2[x][y];
		}
	}


	// do processing
	// float dd = (2*d+1)*(2*d+1);
	for( int x = 0; x < w; x++ ) 
	{
		for( int y = 0; y < h; y++ ) 
		{
			// get bounding box (2d+1)x(2d+1) around (x,y)
			int x0 = x - d-1, x1 = x + d;
			int y0 = y - d-1, y1 = y + d;
			if( x0 <  0 ) { x0 = 0;   }
			if( x1 >= w ) { x1 = w-1; }
			if( y0 <  0 ) { y0 = 0;   }
			if( y1 >= h ) { y1 = h-1; }
			
			float dd = (x1-x0)*(y1-y0);

			// compute mean and variance
			float m = (float)(ii[x1][y1] + ii[x0][y0] - ii[x1][y0] - ii[x0][y1]) / dd;
			float v = (float)(ii2[x1][y1] + ii2[x0][y0] - ii2[x1][y0] - ii2[x0][y1]) / dd  -  (m*m);

			float sigma = 255 * sqrt(v);
			if( sigma < sigma_min ) { sigma = sigma_min; }
				
			mean.at<uchar>(y,x) = saturate_cast<uchar>(255 * m);
			sd.at<uchar>(y,x)   = saturate_cast<uchar>(sigma);
		}
	}

	// blur mean and sd
 	GaussianBlur( mean, mean, Size(21, 21), 0, 0 );
	GaussianBlur(   sd,   sd, Size(21, 21), 0, 0 );

	_lides_mem_counter -= (4.0 * ff);
	cout<< "LIDES MEM<< "<< _lides_mem_counter<< endl;


	// cleaning
	for( int x = 0; x < w; x++ )
	{
		delete [] ii[x];
		delete [] ii2[x];
		delete [] s[x];
		delete [] s2[x];
	}
	delete [] ii;
	delete [] ii2;
	delete [] s;
	delete [] s2;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void local_enhance_gauss( Mat& img, Mat& mean, Mat& sd, Mat& out )
{
	for( int x = 0; x < img.cols; x++ ) 
	{
		for( int y = 0; y < img.rows; y++ ) 
		{
			float f = 0.5 * (1 + erf( (float)(img.at<uchar>(y,x) - mean.at<uchar>(y,x))/
						  (sqrt(2)*(sd.at<uchar>(y,x))) ));
			
			unsigned char v = saturate_cast<uchar>(255 * f);
			out.at<uchar>(y,x) = v;
		}
	}	
}

void local_enhance_laplace( Mat& img, Mat& mean, Mat& sd, Mat& out )
{
	for( int x = 0; x < img.cols; x++ ) 
	{
		for( int y = 0; y < img.rows; y++ ) 
		{
			int sgn = (img.at<uchar>(y,x) - mean.at<uchar>(y,x) > 0 ? 1 : -1);
			double d = (float)(img.at<uchar>(y,x) - mean.at<uchar>(y,x))/sd.at<uchar>(y,x);

			float f = 0.5 * (1 + sgn *(1 - exp(-sqrt(2)*abs(d))));

			unsigned char v = saturate_cast<uchar>(255 * f);
			out.at<uchar>(y,x) = v;
		}
	}	
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void LIDE_simple( Mat& img, Mat& out, int model, int d, float sigma_min )
{
	Mat mean(img.rows, img.cols, CV_8UC1);
	Mat sd  (img.rows, img.cols, CV_8UC1);

	compute_local_stat( img, d, mean, sd, sigma_min );

	if( model == 0 ) { local_enhance_gauss  ( img, mean, sd, out ); }
	else             { local_enhance_laplace( img, mean, sd, out ); }
}




///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "hm.h"

void local_hm_gauss( Mat& img, Mat& mean, Mat& sd, int n, float* cdf, Mat& out )
{
	for( int x = 0; x < img.cols; x++ ) 
	{
		for( int y = 0; y < img.rows; y++ ) 
		{
			float f = 0.5 * (1 + erf( (float)(img.at<uchar>(y,x) - mean.at<uchar>(y,x))/
						  (sqrt(2)*sd.at<uchar>(y,x)) ) );

			int g = inverse_cdf( n, cdf, f );

			unsigned char v = saturate_cast<uchar>(g);
			out.at<uchar>(y,x) = v;
		}
	}	
}

void local_hm_laplace( Mat& img, Mat& mean, Mat& sd, int n, float* cdf, Mat& out )
{
	for( int x = 0; x < img.cols; x++ ) 
	{
		for( int y = 0; y < img.rows; y++ ) 
		{
			int sgn = (img.at<uchar>(y,x) - mean.at<uchar>(y,x) > 0 ? 1 : -1);
			double d = (float)(img.at<uchar>(y,x) - mean.at<uchar>(y,x))/sd.at<uchar>(y,x);

			float f = 0.5 * (1 + sgn *(1 - exp(-sqrt(2)*abs(d))));

			int g = inverse_cdf( n, cdf, f );

			unsigned char v = saturate_cast<uchar>(g);
			out.at<uchar>(y,x) = v;
		}
	}	
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void LIDEHM_simple( Mat& img, Mat& out, int model, int d, float sigma_min, int n, float* hist )
{
	// allocate tmp var
	float* cdf = new float [n];
	build_cdf( n, hist, cdf );

	Mat mean(img.rows, img.cols, CV_8UC1);
	Mat sd  (img.rows, img.cols, CV_8UC1);

	compute_local_stat( img, d, mean, sd, sigma_min );

	if( model == 0 ) { local_hm_gauss  ( img, mean, sd, n, cdf, out ); }
	else             { local_hm_laplace( img, mean, sd, n, cdf, out ); }

	// cleaning
	delete [] cdf;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "bpheme.h"
#include "fhsabp.h"

void LIDE_simple( Mat& img, Mat& out, int model, int d, float sigma_min, int target_distrib )
{
	if( target_distrib == 0 ) // normal
	{
		LIDE_simple( img, out, model, d, sigma_min );
	}
	else 
	{
		// compute average intensity
		float mu = 0.0;
		for( int x = 0; x < img.cols; x++ )
		{
			for( int y = 0; y < img.rows; y++ )
			{
				mu += img.at<uchar>(y,x);
			}
		}
		mu = (float)mu/(img.cols*img.rows);

		// build target distribution
		float* target = new float [256];
		
		switch( target_distrib )
		{
		case 1 : build_target_distrib_bpheme( mu, 256, target ); break;
		case 2 : build_target_distrib_fhsabp( mu, target );      break;
		default : // method 0 
			for( int i = 0; i < 256; i++ )
			{
				target[i] = (float)1.0/256.0;
			}
		}

		LIDEHM_simple( img, out, model, d, sigma_min, 256, target );


		delete [] target;
	}
}
