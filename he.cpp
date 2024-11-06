


#include <cmath>
#include <iostream>
using namespace std;

#include "he.h"
using namespace cv;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float compute_hist( Mat& img, float* hist )
{
	for( int k = 0; k < 256; k++ )
	{
		hist[k] = 0;
	}
	float sum = 0.0;
	for( int x = 0; x < img.cols; x++ )
	{
		for( int y = 0; y < img.rows; y++ )
		{
			int k = img.at<uchar>(y,x);
			hist[k]++;

			sum += (float)k/255.0;
		}
	}
	float w = (float)1.0/(img.cols * img.rows);
	for( int k = 0; k < 256; k++ )
	{
		hist[k] *= w;
	}
	float mean = sum * w * 255.0;

	return mean;
}


float grayscale_remap( Mat& img, int* newval, Mat& out )
{
	float sum = 0.0;
	for( int x = 0; x < img.cols; x++ )
	{
		for( int y = 0; y < img.rows; y++ )
		{
			unsigned char newv = saturate_cast<uchar>(newval[(int)img.at<uchar>(y, x)]);
			out.at<uchar>(y, x) = newv;

			sum += newv;
		}
	}
	float mean = (float)sum/(img.rows * img.cols);

	return mean;
}

void equalize_hist_interval( int beg, int end, float *hist, int *newval )
{
	// allocate tmp vars for cumulative score
	int nlevel = end - beg;
	float *cdf = new float [nlevel];

	float sum = 0.0;
	for( int i = beg; i < end; i++ )
	{
		sum += hist[i];
	}

	int min_cdf = -1, min_idx = -1;
	for( int i = beg; i < end; i++ )
	{
		cdf[i-beg] = ((float)hist[i]/sum + (i == beg ? 0 : cdf[i-1-beg]));

		if( cdf[i-beg]>0 && min_idx==-1 )
		{
			min_idx = i;
			min_cdf = cdf[i-beg];
		}
	}
	if( min_idx==-1 )
	{
		min_idx = beg;
		min_cdf = 0;
	}
	

	// compute new level using histogram-equalization formula
	//cout<< "[ "<< beg<< " "<< end<< "] "<< cdf[end-1-beg]<< " ";

	float d = 1-min_cdf;
	for( int i = beg; i < end; i++ )
	{
	  if( d<1e-8 )
	    {
	      newval[i] = i;
	    }
	  else
	    {
	      newval[i] = beg + (i < min_idx ? 0 : (int)round( nlevel * (float)(cdf[i-beg] - min_cdf)/d ));
	      if( newval[i] > end ) { newval[i] = end; }
		
	    }
	  //	  cout<< "\t"<< i<< " "<< newval[i]<< endl;
	}


	// cleaning
	delete [] cdf;
}


void equalize_hist_interval( int L, float* hist, int M, int* thresholds, int* newval )
{
	for( int i = 0; i < M; i++ )
	{
		int beg = (i==0 ? 0 : thresholds[i-1]);
		int end = (i==M-1 ? 256 : thresholds[i]);
		equalize_hist_interval( beg, end, hist, newval );
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
 * Normal histogram equalization mapping
 */
void equalize_hist0( int nlevel, float *hist, int *newval )
{
	// allocate tmp vars for cumulative score
	float *cdf = new float [nlevel];

	int min_cdf = -1, min_idx = -1;
	for( int i = 0; i < nlevel; i++ )
	{
		cdf[i] = (hist[i] + (i == 0 ? 0 : cdf[i-1]));

		if( cdf[i]>0 && min_idx==-1 )
		{
			min_idx = i;
			min_cdf = cdf[i];
		}
	}
	if( min_idx==-1 )
	{
		min_idx = 0;
		min_cdf = 0;
	}
	

	// compute new level using histogram-equalization formula
	for( int i = 0; i < nlevel; i++ )
	{
		newval[i] = (i < min_idx ? 0 : (int)round( nlevel * (float)(cdf[i] - min_cdf)/(1 - min_cdf) ));
		if( newval[i] > 255 ) { newval[i] = 255; }
	}


	// cleaning
	delete [] cdf;
}

void equalize_hist( int nlevel, float *hist, int *newval )
{
	// allocate tmp vars for cumulative score
	float *cdf = new float [nlevel];

	for( int i = 0; i < nlevel; i++ )
	{
		cdf[i] = (hist[i] + (i == 0 ? 0 : cdf[i-1]));
	}
	
	// compute new level using histogram-equalization formula
	for( int i = 0; i < nlevel; i++ )
	{
//		newval[i] = (int)round( nlevel * cdf[i]);
		newval[i] = (int)( nlevel * cdf[i]);
//		if( newval[i] > 255 ) { newval[i] = 255; }
	}


	// cleaning
	delete [] cdf;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void HE( Mat& img, Mat& out )
{
	// allocate tmp vars
	float* hist   = new float [256];
	int*   newval = new int   [256];

	compute_hist( img, hist );

	equalize_hist( 256, hist, newval );

	for( int x = 0; x < img.cols; x++ )
	{
		for( int y = 0; y < img.rows; y++ )
		{
			unsigned char newv = saturate_cast<uchar>(newval[(int)img.at<uchar>(y, x)]);
			out.at<uchar>(y, x) = newv;
		}
	}

	// cleaning
	delete [] hist;
	delete [] newval;
}

