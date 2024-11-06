


#include <cmath>
#include <iostream>
using namespace std;

#include "he.h"
#include "bpheme.h"
#include "hm.h"
using namespace cv;


//////////////////////////////////////////////////////////////////////////////////////////////

void build_target_distrib_bpheme( int m, int nbin, float* f )
{
	// allocate tmp vars
	float* mu = new float [401];
	for( int l = 0; l < 401; l++ )
	{
		float lambda = -100 + l*0.5;

		mu[l] = (lambda == 0 ? 0.5 : (float)(lambda-1)/lambda + (float)1.0/(exp(lambda)-1));
	}
	

	float lambda = 0;
	if( m == 127 )
	{ 
		lambda = 0; 
	}
	else
	{
		float mm = (float)m/255.0;

		int l = 0;
		while( l<401 && mu[l]<mm ) { l++; }
		if( l>=401 ) { l = 400; }

		lambda = -100 + l*0.5;
	}


	if( lambda == 0 )
	{
		float u = (float)1.0/nbin;
		for( int i = 0; i < nbin; i++ )
		{
			f[i] = u;
		}
	}
	else
	{
		float sum = 0;
		for( int i = 0; i < nbin; i++ )
		{
			float s = (float)i/nbin;
			float g = lambda * (float)exp(lambda*s) / (exp(lambda)-1);

			f[i] = g;
			sum += g;
		}
		for( int i = 0; i < nbin; i++ )
		{
			f[i] = (float)f[i]/sum;
			// cout<< i<< " "<< f[i]<< endl;
		}
	}

	// cleaning
	delete [] mu;
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void BPHEME( Mat& img, Mat& out )
{
	// allocate tmp vars
	float* hist   = new float [256];
	float* target = new float [256];
	int*   newval = new int   [256];

	float m = compute_hist( img, hist );
	cerr<< "m = "<< m<< endl;

	int nbin = 256;
	build_target_distrib_bpheme( m, nbin, target );

	histogram_matching( 256, hist, target, newval );

	float mm = grayscale_remap( img, newval, out );

	// rescale
	float r = (float)m/mm;
	for( int x = 0; x < img.cols; x++ )
	{
		for( int y = 0; y < img.rows; y++ )
		{
			out.at<uchar>(y, x) = saturate_cast<uchar>(out.at<uchar>(y, x) * r);
		}
	}

	// cleaning
	delete [] hist;
	delete [] target;
	delete [] newval;
}



/*


	{
		Mat out (258, 258, CV_8UC3);

		float maxv = 0;
		for( int i = 0; i < 256; i++ )
		{
			cout<< i<< " "<< target[i]<< endl;

			if( maxv < target[i] ) 
			{
				maxv = target[i];
			}
		}
		maxv = (float)7.0/256.0;

		rectangle( out, Point(0,0), Point(258,258), CV_RGB(255,255,255), CV_FILLED);

		for( int i = 0; i < 256; i++ )
		{
			int y = 1 + (int)(256*(float)target[i]/maxv);
		
			line( out, Point(1+i, 257), Point(1+i, 257-y), CV_RGB(0,0,0) );
		}
	
		imwrite( "bpheme.target.png", out );

		cerr<< "OK"<< endl;
		exit(0);
	}

	for( int i = 0; i < 256; i++ )
	{
		cout<< i<< " "<< target[i]<< endl;
	}
	exit(0);
*/
