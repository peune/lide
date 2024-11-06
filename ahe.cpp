

#include <cmath>
#include <iostream>
using namespace std;

#include "he.h"
#include "ahe.h"
using namespace cv;



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
 * Adaptive histogram equalization. 
 * Idea: use 256 integral image to speed up the calculation of local histogram around each pixel
 */

FastLocalHist::FastLocalHist( int new_w, int new_h, int new_nlevel )
{
	nlevel = new_nlevel;
	w      = new_w;
	h      = new_h;

	ii = new int** [nlevel];
	for( int k = 0; k < nlevel; k++ )
	{
		ii[k] = new int* [w];
		for( int x = 0; x < w; x++ )
		{
			ii[k][x] = new int [h];
		}
	}
}
FastLocalHist::~FastLocalHist()
{
	for( int k = 0; k < nlevel; k++ )
	{
		for( int x = 0; x < w; x++ )
		{
			delete [] ii[k][x];
		}
		delete [] ii[k];
	}
	delete [] ii;

}

void FastLocalHist::init( Mat& img, int x0, int y0 )
{
	// allocate tmp vars
	int **s = new int* [w];
	for( int x = 0; x < w; x++ ) 
	{
		s[x] = new int [h];
	}


	// compute integral image for each grayscale level, and their sum
	for( int k = 0; k < nlevel; k++ ) 
	{
		for( int x = 0; x < w; x++ )
		{
			for( int y = 0; y < h; y++ )
			{
				int v = 0;
				if( x0+x<img.cols && y0+y<img.rows && img.at<uchar>(y0+y,x0+x)==k )
				{
					v = 1;
				}
								
				s[x][y]     = (y == 0 ? 0 :  s[x][y-1]    + v);
				ii[k][x][y] = (x == 0 ? 0 : ii[k][x-1][y] + s[x][y]);
			}
		}
	}

	
	// cleaning
	for( int x = 0; x < w; x++ )
	{
		delete [] s[x];
	}
	delete [] s;
}

void FastLocalHist::compute_hist( int x0, int y0, int x1, int y1, float clipping, float* hist )
{
	float s = 0.0;
	for( int i = 0; i < nlevel; i++ ) 
	{
		/* assumed that it was already checked!!
		int v = ( (x1 <  w  && y1 <  h  ? ii[i][x1][y1] : 0) +
			  (0  <= x0 && 0  <= y0 ? ii[i][x0][y0] : 0) -
			  (x1 <  w  && 0  <= y0 ? ii[i][x1][y0] : 0) -
			  (0  <= x0 && y1 <  h  ? ii[i][x0][y1] : 0) );
		*/

		int v = ii[i][x1][y1] + ii[i][x0][y0] - ii[i][x1][y0] - ii[i][x0][y1];

		hist[i] = ((1-clipping) * v);
		s      += (clipping * v);
	}
	s = (float)s/nlevel;

	float sum = 0;
	for( int i = 0; i < nlevel; i++ ) 
	{
		hist[i] += s;
		sum += hist[i];
	}
	for( int i = 0; i < nlevel; i++ ) 
	{
		hist[i] = (float)hist[i]/sum;
	}
}


//
// Do local histogram equalization within sub-window (bx,by,w,h)
//
void equalize_local_hist( Mat& img, int bx, int by, int w, int h,
			  int d, FastLocalHist& flh, float clipping, 
			  Mat& out )
{
	// allocate tmp vars
	float* hist   = new float [256];
	int*   newval = new int   [256];

	flh.init( img, bx, by );
	cout<< "MEM NOW = "<< _ahe_mem_counter<< endl;

	int W = img.cols, H = img.rows;

	int beginx = (bx == 0 ? 0 : d);
	int beginy = (by == 0 ? 0 : d);

	for( int x = beginx; x < w; x++ )
	{
		for( int y = beginy; y < h; y++ )
		{
			int xx = bx+x, yy = by+y;
			
			if( xx<W && yy<H )
			{
				// local window (2d+1)*(2d+1) around (x,y)
				int x0 = x - d-1, x1 = x + d;
				int y0 = y - d-1, y1 = y + d;
				if( x0 <  0  ) { x0 = 0;   }
				if( w  <= x1 ) { x1 = w-1; }
				if( y0 <  0  ) { y0 = 0;   }
				if( h  <= y1 ) { y1 = h-1; }

				flh.compute_hist( x0, y0, x1, y1, clipping, hist );
		
				equalize_hist( 256, hist, newval );
			
				
				unsigned char newv = saturate_cast<uchar>(newval[(int)img.at<uchar>(yy, xx)]);
				out.at<uchar>(yy, xx) = newv;
			}
		}
	}


	// cleaning
	delete [] hist;
	delete [] newval;
}

float get_mean( Mat& img )
{
	float sum = 0.0;
	for( int x = 0; x < img.cols; x++ )
	{
		for( int y = 0; y < img.rows; y++ )
		{
			sum += (float)img.at<uchar>(y,x)/255.0;
		}
	}
	float mean = 255 * (float)sum / (img.cols*img.rows);

	return mean;
}

void AHE( Mat& img, Mat& out, int d, float clipping, bool rescale_intensity )
{
/*
	int subw = (3*d < img.cols ? 3*d : img.cols);
	int subh = (3*d < img.rows ? 3*d : img.rows);

	FastLocalHist flh(subw, subh);

	for( int x = 0; x < img.cols; x = (x+subw-2*d>img.cols ? img.cols : x+subw-2*d) )
	{
		for( int y = 0; y < img.rows; y = (y+subh-2*d>img.rows ? img.rows : y+subh-2*d) )
		{
			equalize_local_hist( img, x, y, subw, subh, d, flh, out );
		}
	}
*/
	FastLocalHist flh( img.cols, img.rows ); 
	equalize_local_hist( img, 0, 0, img.cols, img.rows, d, flh, clipping, out );


	if( rescale_intensity )
	{
		float m0 = get_mean( img );
		float m1 = get_mean( out );
		float r = (float)m0/m1;
		
		for( int x = 0; x < img.cols; x++ )
		{
			for( int y = 0; y < img.rows; y++ )
			{
				out.at<uchar>(y,x) = saturate_cast<uchar>( out.at<uchar>(y,x) * r );
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//
// Do local histogram matching within sub-window (bx,by,w,h)
//
#include "hm.h"

void matching_local_hist( Mat& img, int bx, int by, int w, int h,
			  int d, FastLocalHist& flh, float clipping, 
			  float* target, 
			  Mat& out )
{
	// allocate tmp vars
	float* hist   = new float [256];
	int*   newval = new int   [256];

	flh.init( img, bx, by );

	int W = img.cols, H = img.rows;

	int beginx = (bx == 0 ? 0 : d);
	int beginy = (by == 0 ? 0 : d);

	for( int x = beginx; x < w; x++ )
	{
		for( int y = beginy; y < h; y++ )
		{
			int xx = bx+x, yy = by+y;
			
			if( xx<W && yy<H )
			{
				// local window (2d+1)*(2d+1) around (x,y)
				int x0 = x - d-1, x1 = x + d;
				int y0 = y - d-1, y1 = y + d;
				if( x0 <  0  ) { x0 = 0;   }
				if( w  <= x1 ) { x1 = w-1; }
				if( y0 <  0  ) { y0 = 0;   }
				if( h  <= y1 ) { y1 = h-1; }

				flh.compute_hist( x0, y0, x1, y1, clipping, hist );
		
				histogram_matching( 256, hist, target, newval );
				
				unsigned char newv = saturate_cast<uchar>(newval[(int)img.at<uchar>(yy, xx)]);
				out.at<uchar>(yy, xx) = newv;
			}
		}
	}


	// cleaning
	delete [] hist;
	delete [] newval;
}


void AHEHM( Mat& img, Mat& out, int d, float clipping, float* target )
{
	FastLocalHist flh( img.cols, img.rows ); 
	matching_local_hist( img, 0, 0, img.cols, img.rows, d, flh, clipping, target, out );
}
