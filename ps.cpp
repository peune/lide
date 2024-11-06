

#include <cmath>
#include <iostream>
using namespace std;

#include "he.h"
#include "ps.h"

#include <opencv2/opencv.hpp>
using namespace cv;


///////////////////////////////////////////////////////////////////////////

float compute_local_mean( Mat& img, int d, Mat& mean )
{
	// allocate tmp vars
	int w = img.cols;
	int h = img.rows;
	float **ii  = new float* [w];
	float **s   = new float* [w];
	for( int x = 0; x < w; x++ ) 
	{
		ii[x]  = new float [h];
		s[x]   = new float [h];
	}


	// compute integral images
	for( int x = 0; x < w; x++ ) 
	{
		for( int y = 0; y < h; y++ ) 
		{
			int   g = img.at<unsigned char>(y,x);
			float p = (float)g/255.0;

			s[x][y]  = (y==0 ? 0 : s[x][y-1])  + p;
			ii[x][y] = (x==0 ? 0 : ii[x-1][y]) + s[x][y];
		}
	}


	float minv = HUGE_VAL, maxv = -HUGE_VAL;

	// do processing
//	int d2 = d/2; 
	for( int x = 0; x < w; x++ ) 
	{
		for( int y = 0; y < h; y++ ) 
		{
			// get bounding box
			int x0 = x - d-1, x1 = x + d;
			int y0 = y - d-1, y1 = y + d;
			if( x0 <  0 ) { x0 = 0;   }
			if( x1 >= w ) { x1 = w-1; }
			if( y0 <  0 ) { y0 = 0;   }
			if( y1 >= h ) { y1 = h-1; }

			float dd = (x1-x0)*(y1-y0);

			// compute mean and variance
			float m = ( (x1 < w && y1 < h ? ii[x1][y1] : 0)
				     + (0 <= x0 && 0 <= y0 ? ii[x0][y0] : 0)
				     - (x1 < w  && 0 <= y0 ? ii[x1][y0] : 0)
				     - (0 <= x0 && y1 < h  ? ii[x0][y1] : 0) ) / dd;
			
			if( minv > m ) { minv = m; }
			if( maxv < m ) { maxv = m; }
			
			mean.at<uchar>(y,x) = saturate_cast<uchar>(255 * m);
		}
	}
	// cerr<< 255*minv<< " "<< 255*maxv<< endl;

	float mm = 255 * (float)(ii[w-1][h-1] + ii[0][0] - ii[0][h-1] - ii[w-1][0]) / (w*h);


	// cleaning
	for( int x = 0; x < w; x++ )
	{
		delete [] ii[x];
		delete [] s[x];
	}
	delete [] ii;
	delete [] s;

	return mm;
}

///////////////////////////////////////////////////////////////////////////
#include "sort.h"

float compute_ps1( Mat& img, int d, float* val, int* idx )
{
	Mat mean( img.rows, img.cols, CV_8UC1 );
	
	float mu = compute_local_mean( img, d, mean );
	cerr<< mu<< endl;

	for( int y = 0; y < img.rows; y++ )
	{
		for( int x = 0; x < img.cols; x++ )
		{
			int k = x + y*img.cols;
			val[k] = (float)(img.at<uchar>(y,x) * 256 + mean.at<uchar>(y,x)) / 65025.0;

			idx[k] = k;
		}
	}
	cerr<< "ICICIC"<< endl;

	return mu;
}

////////////////////////////////////////////////////////////////////////////////////////

float compute_ps2( Mat& img, int d, float* val, int* idx )
{
	Mat mean( img.rows, img.cols, CV_8UC1 );
	
	float mu = compute_local_mean( img, d, mean );

	for( int y = 0; y < img.rows; y++ )
	{
		for( int x = 0; x < img.cols; x++ )
		{
			int k = x + y*img.cols;
			// val[k] = 0.1*(float)img.at<uchar>(y,x) / (mean.at<uchar>(y,x) + 1e-8) + (float)(img.at<uchar>(y,x) * mean.at<uchar>(y,x)) / 65025.0;
//img.at<uchar>(y,x)/255.0;

			float a = (float)img.at<uchar>(y,x) / 255.0;
			float b = (float)mean.at<uchar>(y,x) / 255.0;
			
			val[k] = b; // 
			// val[k] = (float)(a*b)/(a+b+1e-8);
			// val[k] = (float)a/(b+1e-8);

			idx[k] = k;
		}
	}

	return mu;
}

////////////////////////////////////////////////////////////////////////////////////////

#include "lide_simple.h"
float compute_ps3( Mat& img, int d, float* val, int* idx )
{
	Mat mean( img.rows, img.cols, CV_8UC1 );
	Mat sd( img.rows, img.cols, CV_8UC1 );
	
	compute_local_stat( img, d, mean, sd, 5 );

	for( int y = 0; y < img.rows; y++ )
	{
		for( int x = 0; x < img.cols; x++ )
		{
			int k = x + y*img.cols;

			float a = (float)img.at<uchar>(y,x) / 255.0;
			float b = (float)mean.at<uchar>(y,x) / 255.0;
			float c = (float)sd.at<uchar>(y,x) / 255.0;
			
			val[k] = (float)(a-b)/c;

			idx[k] = k;
		}
	}

	return 0;
}

////////////////////////////////////////////////////////////////////////////////////////


#include "hegmm.h"
#include "mhe.h"
float compute_ps3old( Mat& img, int d, float* val, int* idx )
{
	Mat mean( img.rows, img.cols, CV_8UC1 );
	
	float mu = compute_local_mean( img, d, mean );

	int maxx=0, maxy=0, maxv=0;
	for( int y = 0; y < img.rows; y++ )
	{
		for( int x = 0; x < img.cols; x++ )
		{
			int v = img.at<uchar>(y,x);
			if( maxv < v ) 
			{
				maxv = v;
				maxx = x;
				maxy = y;
			}
		}
	}

	for( int y = 0; y < img.rows; y++ )
	{
		for( int x = 0; x < img.cols; x++ )
		{
			float dx = x-maxx, dy = y-maxy;
			float d = sqrt(dx*dx + dy*dy);

			int k = x + y*img.cols;
			val[k] = (float)img.at<uchar>(y,x) / (mean.at<uchar>(y,x) + 1e-8) + 5*mean.at<uchar>(y,x)/(d + 255.0);
			idx[k] = k;
		}
	}

	return mu;
}

float compute_ps3d( Mat& img, int d, float* val, int* idx )
{
	// allocate tmp vars
	float* hist = new float [256];
	compute_hist( img, hist );

	for( int i = 0; i < 256; i++ )
	{
//		cout<< i<< " "<< hist[i]<< endl;
	}

	int* thresholds = new int [256];
	int M = mhe_partition( 256, hist, thresholds, 5, 8, 0.8 );

	Mat tmp(img.rows, img.cols, CV_8UC1);
	for( int x = 0; x < img.cols; x++ )
	{
		for( int y = 0; y < img.rows; y++ )
		{
			int g = img.at<uchar>(y,x);
			int i = 0;
			while( i<=M && thresholds[i]<g ) { i++; }
			
			tmp.at<uchar>(y,x) = (i==0 ? 0 : thresholds[i-1]);
		}
	}
//	imshow("or", img);
//	imshow("im", tmp);
//	waitKey(-1);


	Mat mean( img.rows, img.cols, CV_8UC1 );
	
	float mmu = compute_local_mean( img, d, mean );


	for( int y = 0; y < img.rows; y++ )
	{
		for( int x = 0; x < img.cols; x++ )
		{
			int g = img.at<uchar>(y,x);

			int i = 0;
			while( i<=M && thresholds[i]<g ) { i++; }
			int b = (i==0 ? 0 : thresholds[i-1]);
			// int e = (i==M ? 255 : thresholds[i]);
			float f = b; //  * (float)(i-b)/(e-b);
			
			
			int k = x + y*img.cols;
			val[k] = (float)g / (mean.at<uchar>(y,x) + 1e-8) + 0.0001*f;
			idx[k] = k;
		}
	}

	// cleaning
	delete [] hist;


	return mmu;
}

////////////////////////////////////////////////////////////////////////////////////////

float return_mean( int w, int h, float** ii,
		   int x, int y, int d2 )
{
	// get bounding box
	int x0 = x - d2, x1 = x + d2;
	int y0 = y - d2, y1 = y + d2;
	if( x0 <  0 ) { x0 = 0;   }
	if( x1 >= w ) { x1 = w-1; }
	if( y0 <  0 ) { y0 = 0;   }
	if( y1 >= h ) { y1 = h-1; }

	float dd = (x1-x0)*(y1-y0);

	// compute mean and variance
	float m = ( (x1 < w && y1 < h ? ii[x1][y1] : 0)
		    + (0 <= x0 && 0 <= y0 ? ii[x0][y0] : 0)
		    - (x1 < w  && 0 <= y0 ? ii[x1][y0] : 0)
		    - (0 <= x0 && y1 < h  ? ii[x0][y1] : 0) ) / dd;

	return m;
}
float compute_ps3a( Mat& img, int d, float* val, int* idx )
{
	// allocate tmp vars
	int w = img.cols;
	int h = img.rows;
	float **ii  = new float* [w];
	float **s   = new float* [w];
	for( int x = 0; x < w; x++ ) 
	{
		ii[x]  = new float [h];
		s[x]   = new float [h];
	}


	// compute integral images
	for( int x = 0; x < w; x++ ) 
	{
		for( int y = 0; y < h; y++ ) 
		{
			int   g = img.at<unsigned char>(y,x);
			float p = (float)g/255.0;

			s[x][y]  = (y==0 ? 0 : s[x][y-1])  + p;
			ii[x][y] = (x==0 ? 0 : ii[x-1][y]) + s[x][y];
		}
	}


	// float minv = HUGE_VAL, maxv = -HUGE_VAL;

	// do processing
	int d2 = d/2, d4 = d/4; 
	for( int x = 0; x < w; x++ ) 
	{
		for( int y = 0; y < h; y++ ) 
		{
			float m2 = return_mean( w, h, ii, x, y, d2 );
			float m4 = return_mean( w, h, ii, x, y, d4 );

			float g = (float)img.at<uchar>(y,x) / 255.0; 
			
			int k = x + y*img.cols;
			val[k] = g/(m4 + 1e-8)   +   m4/(m2 + 1e-8);  
			idx[k] = k;

		}
	}

	float mm = 255 * (float)(ii[w-1][h-1] + ii[0][0] - ii[0][h-1] - ii[w-1][0]) / (w*h);


	// cleaning
	for( int x = 0; x < w; x++ )
	{
		delete [] ii[x];
		delete [] s[x];
	}
	delete [] ii;
	delete [] s;

	return mm;
}

////////////////////////////////////////////////////////////////////////////////////////

float compute_ps4( Mat& img, int d, float* val, int* idx )
{
	Mat mean( img.rows, img.cols, CV_8UC1 );
	
	float mu = compute_local_mean( img, d, mean );

//	float mu = 0.0;
	for( int y = 0; y < img.rows; y++ )
	{
		for( int x = 0; x < img.cols; x++ )
		{
			float v = 0;
			if( 1<=x && x<img.cols-1 && 1<=y && y<img.rows-1 )
			{
				float gx = 0.25 * ( (img.at<uchar>(y-1, x+1) - img.at<uchar>(y-1, x-1))   + 
						    (img.at<uchar>(y  , x+1) - img.at<uchar>(y  , x-1))*2 +
						    (img.at<uchar>(y-1, x+1) - img.at<uchar>(y-1, x-1)) ); 

				float gy = 0.25 * ( (img.at<uchar>(y+1, x-1) - img.at<uchar>(y-1, x-1))   + 
						    (img.at<uchar>(y+1, x  ) - img.at<uchar>(y-1, x  ))*2 +
						    (img.at<uchar>(y+1, x+1) - img.at<uchar>(y-1, x+1)) );

				v = sqrt( gx*gx + gy*gy );
			}
			v += mean.at<uchar>(y,x);

			int k = x + y*img.cols;
			val[k] = v;
			idx[k] = k;

//			mu += img.at<uchar>(y,x);
		}
	}
//	mu = (float)mu/(img.rows*img.cols);

	return mu;
}


////////////////////////////////////////////////////////////////////////////////////////

float compute_ps5( Mat& img, int d, float* val, int* idx )
{
	// allocate tmp vars
	int w = img.cols;
	int h = img.rows;
	float **iixx  = new float* [w];
	float **sxx   = new float* [w];
	float **iiyy  = new float* [w];
	float **syy   = new float* [w];
	float **iixy  = new float* [w];
	float **sxy   = new float* [w];
	for( int x = 0; x < w; x++ ) 
	{
		iixx[x]  = new float [h];
		sxx[x]   = new float [h];
		iiyy[x]  = new float [h];
		syy[x]   = new float [h];
		iixy[x]  = new float [h];
		sxy[x]   = new float [h];
	}


	// compute integral images
	for( int x = 0; x < w; x++ ) 
	{
		for( int y = 0; y < h; y++ ) 
		{
			float gx=0, gy=0;
			if( 1<=x && x<img.cols-1 && 1<=y && y<img.rows-1 )
			{
				gx = 0.25 * ( (img.at<uchar>(y-1, x+1) - img.at<uchar>(y-1, x-1))   + 
					      (img.at<uchar>(y  , x+1) - img.at<uchar>(y  , x-1))*2 +
					      (img.at<uchar>(y-1, x+1) - img.at<uchar>(y-1, x-1)) ); 
				
				gy = 0.25 * ( (img.at<uchar>(y+1, x-1) - img.at<uchar>(y-1, x-1))   + 
					      (img.at<uchar>(y+1, x  ) - img.at<uchar>(y-1, x  ))*2 +
					      (img.at<uchar>(y+1, x+1) - img.at<uchar>(y-1, x+1)) );

			}
		
			float xx = gx*gx;
			float yy = gy*gy;
			float xy = gx*gy;

			sxx[x][y]  = (y==0 ? 0 : sxx[x][y-1])  + xx;
			iixx[x][y] = (x==0 ? 0 : iixx[x-1][y]) + sxx[x][y];

			syy[x][y]  = (y==0 ? 0 : syy[x][y-1])  + yy;
			iiyy[x][y] = (x==0 ? 0 : iiyy[x-1][y]) + syy[x][y];

			sxy[x][y]  = (y==0 ? 0 : sxy[x][y-1])  + xy;
			iixy[x][y] = (x==0 ? 0 : iixy[x-1][y]) + sxy[x][y];
		}
	}



	int d2 = 2;// d/2;
	float mu = 0.0;
	for( int y = 0; y < img.rows; y++ )
	{
		for( int x = 0; x < img.cols; x++ )
		{
			// get bounding box
			int x0 = x - d2, x1 = x + d2;
			int y0 = y - d2, y1 = y + d2;
			if( x0 <  0 ) { x0 = 0;   }
			if( x1 >= w ) { x1 = w-1; }
			if( y0 <  0 ) { y0 = 0;   }
			if( y1 >= h ) { y1 = h-1; }

			float dd = (x1-x0)*(y1-y0);

			// compute mean and variance
			float xx = ( (x1 < w && y1 < h ? iixx[x1][y1] : 0)
				     + (0 <= x0 && 0 <= y0 ? iixx[x0][y0] : 0)
				     - (x1 < w  && 0 <= y0 ? iixx[x1][y0] : 0)
				     - (0 <= x0 && y1 < h  ? iixx[x0][y1] : 0) ) / dd;

			float yy = ( (x1 < w && y1 < h ? iiyy[x1][y1] : 0)
				     + (0 <= x0 && 0 <= y0 ? iiyy[x0][y0] : 0)
				     - (x1 < w  && 0 <= y0 ? iiyy[x1][y0] : 0)
				     - (0 <= x0 && y1 < h  ? iiyy[x0][y1] : 0) ) / dd;

			float xy = ( (x1 < w && y1 < h ? iixy[x1][y1] : 0)
				     + (0 <= x0 && 0 <= y0 ? iixy[x0][y0] : 0)
				     - (x1 < w  && 0 <= y0 ? iixy[x1][y0] : 0)
				     - (0 <= x0 && y1 < h  ? iixy[x0][y1] : 0) ) / dd;
			
			float det = xx*yy - xy*xy;
			float tr  = xx + yy;

			float v = (double)det / (tr + 1e-10);

			int k = x + y*img.cols;
			val[k] = v;
			idx[k] = k;

			mu += img.at<uchar>(y,x);
		}
	}
	mu = (float)mu/(img.rows*img.cols);



	// cleaning
	for( int x = 0; x < w; x++ )
	{
		delete [] iixx[x];
		delete [] sxx[x];
		delete [] iiyy[x];
		delete [] syy[x];
		delete [] iixy[x];
		delete [] sxy[x];
	}
	delete [] iixx;
	delete [] sxx;
	delete [] iiyy;
	delete [] syy;
	delete [] iixy;
	delete [] sxy;




	return mu;
}

////////////////////////////////////////////////////////////////////////////////////////
void harris( Mat& img, Mat& out )
{
	Mat Gx( img.rows, img.cols, CV_32F );
	Mat Gy( img.rows, img.cols, CV_32F );
	Mat tmp( img.rows, img.cols, CV_32F );

	for( int x = 0; x < img.cols; x++ ) 
	{
		for( int y = 0; y < img.rows; y++ ) 
		{
			float gx=0, gy=0;
			if( 1<=x && x<img.cols-1 && 1<=y && y<img.rows-1 )
			{
				gx = 0.25 * ( (img.at<uchar>(y-1, x+1) - img.at<uchar>(y-1, x-1))   + 
					      (img.at<uchar>(y  , x+1) - img.at<uchar>(y  , x-1))*2 +
					      (img.at<uchar>(y-1, x+1) - img.at<uchar>(y-1, x-1)) ); 
				
				gy = 0.25 * ( (img.at<uchar>(y+1, x-1) - img.at<uchar>(y-1, x-1))   + 
					      (img.at<uchar>(y+1, x  ) - img.at<uchar>(y-1, x  ))*2 +
					      (img.at<uchar>(y+1, x+1) - img.at<uchar>(y-1, x+1)) );

			}
			Gx.at<float>(y,x) = gx;
			Gy.at<float>(y,x) = gy;
		}
	}

	rectangle( out, Point(0,0), Point(img.cols, img.rows), CV_RGB(0,0,0) );

	int dx[] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
	int dy[] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };

	float maxv=-HUGE_VAL, minv=HUGE_VAL;
	for( int x = 1; x < img.cols-1; x++ ) 
	{
		for( int y = 1; y < img.rows-1; y++ ) 
		{
			float Ix2=0.0, Iy2=0.0, IxIy=0.0; 
			for( int k = 0; k < 9; k++ )
			{
				Ix2  += (Gx.at<float>(y+dy[k], x+dx[k]) * Gx.at<float>(y+dy[k], x+dx[k]));
				Iy2  += (Gy.at<float>(y+dy[k], x+dx[k]) * Gy.at<float>(y+dy[k], x+dx[k]));
				IxIy += (Gx.at<float>(y+dy[k], x+dx[k]) * Gy.at<float>(y+dy[k], x+dx[k]));
			}

			float det = Ix2*Iy2 - IxIy*IxIy;
			float tr  = Ix2 + Iy2;
			float v   = (float)det/(tr+1e-10);
			
			tmp.at<float>(y,x) = v;
			if( maxv < v ) { maxv = v; }
			if( minv > v ) { minv = v; }
		}
	}
	for( int x = 1; x < img.cols-1; x++ ) 
	{
		for( int y = 1; y < img.rows-1; y++ ) 
		{
			out.at<uchar>(y,x) = saturate_cast<uchar>(255 * (float)(tmp.at<float>(y,x)-minv)/(maxv-minv));
		}
	}

	// imshow("out", out);
	// waitKey(-1);
	// exit(0);
}



////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

void hm_from_sorting( int l, int* idx, float* target, int w, Mat& out )
{
	int t = 0;
	for( int i = 0; i < 256; i++ )
	{
		int end = t + (int)(l * target[i]);
		if( end >= l ) { end = l-1; }

		for( ; t <= end; t++ )
		{
			int y = idx[t] / w;
			int x = idx[t] % w;

			out.at<uchar>(y,x) = i;
		}
	}	
}

////////////////////////////////////////////////////////////////////////////////////////

float compute_psps( Mat& img, float* val, int* idx )
{
	// allocate tmp vars
	int w = img.cols;
	int h = img.rows;
	float **ii  = new float* [w];
	float **s   = new float* [w];
	for( int x = 0; x < w; x++ ) 
	{
		ii[x]  = new float [h];
		s[x]   = new float [h];
	}


	// compute integral images
	for( int x = 0; x < w; x++ ) 
	{
		for( int y = 0; y < h; y++ ) 
		{
			int   g = img.at<unsigned char>(y,x);
			float p = (float)g/255.0;

			s[x][y]  = (y==0 ? 0 : s[x][y-1])  + p;
			ii[x][y] = (x==0 ? 0 : ii[x-1][y]) + s[x][y];
		}
	}

	int tabd[] = { 10, 50, 100, 200 };
	
	float alpha = 1; // 0.9;

	// do processing
	for( int x = 0; x < w; x++ ) 
	{
		for( int y = 0; y < h; y++ ) 
		{
			float p = 0; // (float)img.at<uchar>(y,x) / 255.0;
			float sum = p;

			for( int k = 0; k < 4; k++ )
			{
				int d = tabd[k];

				// get bounding box
				int x0 = x - d-1, x1 = x + d;
				int y0 = y - d-1, y1 = y + d;
				if( x0 <  0 ) { x0 = 0;   }
				if( x1 >= w ) { x1 = w-1; }
				if( y0 <  0 ) { y0 = 0;   }
				if( y1 >= h ) { y1 = h-1; }

				float dd = (x1-x0)*(y1-y0);

				// compute mean
				float m = ( (x1 < w && y1 < h ? ii[x1][y1] : 0)
					    + (0 <= x0 && 0 <= y0 ? ii[x0][y0] : 0)
					    - (x1 < w  && 0 <= y0 ? ii[x1][y0] : 0)
					    - (0 <= x0 && y1 < h  ? ii[x0][y1] : 0) ) / dd;

				sum += alpha*(p-m)/m;
				alpha *= alpha;
			}

			int k = x + y*img.cols;
			val[k] = sum;
			idx[k] = k;
		}
	}



	// cleaning
	for( int x = 0; x < w; x++ )
	{
		delete [] ii[x];
		delete [] s[x];
	}
	delete [] ii;
	delete [] s;



	return 0;
}


float ps_sort( Mat& img, int d, int* idx,
	       int phi )
{
	// allocate tmp vars
	int l = img.cols * img.rows;
	float* val = new float [l];

	// compute transfer function
	float mu = 0.0;
/*
	if     ( phi==1 ) { mu = compute_ps1( img, d, val, idx ); }
	else if( phi==2 ) { mu = compute_ps2( img, d, val, idx ); }
	else if( phi==3 ) { mu = compute_ps3( img, d, val, idx ); }
	else if( phi==4 ) { mu = compute_ps4( img, d, val, idx ); }
	else if( phi==5 ) { mu = compute_ps5( img, d, val, idx ); }
*/
	compute_psps( img, val, idx );

	cerr<< "CCCCC"<< endl;

	// guard against uniform dark area ('0' )
	int beg = 0;
	for( int i = 0; i < l && phi != 3; i++ )
	{
		if( val[i] == 0 )
		{
			float vtmp = val[beg];
			val[beg] = val[i];
			val[i]   = vtmp;

			int itmp = idx[beg];
			idx[beg] = idx[i];
			idx[i]   = itmp;
			
			beg++;
		}
	}

	cerr<< "ooo "<< beg<< " "<< l<< endl;

	double t0 = clock();
	quick_sort_oO( val, idx, beg, l );
	double t1 = clock();
	cerr<< (double)(t1-t0)/CLOCKS_PER_SEC<< endl;

	// cleaning
	delete [] val;

	return mu;
}

////////////////////////////////////////////////////////////////////////////////////////





#include "bpheme.h"
#include "fhsabp.h"
void PS( Mat& img, Mat& out, int d, int phi, int target_distrib )
{
	// allocate tmp vars
	int l = img.rows * img.cols;
	int*   idx    = new int   [l];
	float* target = new float [256];

	float mu = ps_sort( img, d, idx, phi ); // ..
	
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

	hm_from_sorting( l, idx, target, img.cols, out );

	// imshow("im", out );
	// waitKey(-1);

	// cleaning
	delete [] idx;
	delete [] target;
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


float reg_compute_ps2( Mat& img, Mat& V, float lambda, int d, float* val, int* idx, float& minv )
{
	Mat mean( img.rows, img.cols, CV_8UC1 );
	
	float mu = compute_local_mean( img, d, mean );

	minv  = HUGE_VAL;
	float maxv = -HUGE_VAL;
	for( int y = 0; y < img.rows; y++ )
	{
		for( int x = 0; x < img.cols; x++ )
		{
			float v = (float)img.at<uchar>(y,x) / (mean.at<uchar>(y,x) + 1e-8);

			int k = x + y*img.cols;
			val[k] = v + lambda*0.25*V.at<uchar>(y,x);
			idx[k] = k;

			if( minv > v ) { minv = v; }
			if( maxv < v ) { maxv = v; }
		}
	}
	cerr<< maxv<< endl;

	return mu;
}

float reg_ps_sort( Mat& img, int d, int* idx,
		   int phi, 
		   Mat& V, float lambda )
{
	// allocate tmp vars
	int l = img.cols * img.rows;
	float* val = new float [l];

	// compute transfer function
	float minv=0;
	float mu = reg_compute_ps2( img, V, lambda, d, val, idx, minv ); 

	// guard against uniform dark area ('0' )
	int beg = 0;
	for( int i = 0; i < l && phi != 3; i++ )
	{
		val[i] -= minv;

		if( val[i] == 0 )
		{
			float vtmp = val[beg];
			val[beg] = val[i];
			val[i]   = vtmp;

			int itmp = idx[beg];
			idx[beg] = idx[i];
			idx[i]   = itmp;
			
			beg++;
		}
	}

	quick_sort_oO( val, idx, beg, l );

	// cleaning
	delete [] val;

	return mu;
}

int sign( int v ) { return (v < 0 ? -1 : 1); }

void PSreg( Mat& img, Mat& out, int d, int phi, int target_distrib, 
	    float lambda )
{
	// allocate tmp vars
	int l = img.rows * img.cols;
	int*   idx    = new int   [l];
	float* target = new float [256];
	Mat V(img.rows, img.cols, CV_8UC1, Scalar(0) );
	
	for( int i = 0; i < 256; i++ )
	{
		target[i] = (float)1.0/256.0;
	}
	
	imshow("org", img);
	waitKey(-1);


	// init
	float mu = ps_sort( img, d, idx, phi ); // ..

	hm_from_sorting( l, idx, target, img.cols, out );

	char filename[1000];
	for( int iter = 0; iter < 20; iter++ )
	{
		// compute V
		for( int x = 1; x < img.cols-1; x++ )
		{
			for( int y = 1; y < img.rows-1; y++ )
			{
				V.at<uchar>(y,x) = ( sign(out.at<uchar>(y  ,x+1) - out.at<uchar>(y  ,x  )) + 
						     sign(out.at<uchar>(y+1,x  ) - out.at<uchar>(y  ,x  )) + 
						     sign(out.at<uchar>(y  ,x  ) - out.at<uchar>(y  ,x-1)) + 
						     sign(out.at<uchar>(y  ,x  ) - out.at<uchar>(y-1,x  )) ); 
						     
			}
		}

		// update
		mu = reg_ps_sort( img, d, idx, phi, V, lambda );

		hm_from_sorting( l, idx, target, img.cols, out );

		imshow("im", out );
		waitKey(-1);

		sprintf( filename, "out_%.d.png", iter );
		imwrite( filename, out );
	}



	// cleaning
	delete [] idx;
	delete [] target;
}








/////
float compute_ps3_color( Mat& img, Mat& rgb_img, int d, float* val, int* idx )
{
       Mat mean( img.rows, img.cols, CV_8UC1 );
	
	float mu = compute_local_mean( img, d, mean );

	for( int y = 0; y < img.rows; y++ )
	{
		for( int x = 0; x < img.cols; x++ )
		{
			int b = rgb_img.at<Vec3b>(y,x)[0];
			int g = rgb_img.at<Vec3b>(y,x)[1];
			int r = rgb_img.at<Vec3b>(y,x)[2];
			
			float f = 1.0 - (float)r/(r+g+b);

			int k = x + y*img.cols;
			val[k] = 10*log(f+1e-10) +  (float)img.at<uchar>(y,x) / (mean.at<uchar>(y,x) + 1e-8) + mean.at<uchar>(y,x)/255.0;
			idx[k] = k;
		}
	}

	return mu;
}

float ps_sort_color( Mat& img, Mat& rgb_img, int d, int* idx,
		     int phi ){
	// allocate tmp vars
	int l = img.cols * img.rows;
	float* val = new float [l];

	// compute transfer function
	float mu = compute_ps3_color( img, rgb_img, d, val, idx );

	// guard against uniform dark area ('0' )
	int beg = 0;
	for( int i = 0; i < l && phi != 3; i++ )
	{
		if( val[i] == 0 )
		{
			float vtmp = val[beg];
			val[beg] = val[i];
			val[i]   = vtmp;

			int itmp = idx[beg];
			idx[beg] = idx[i];
			idx[i]   = itmp;
			
			beg++;
		}
	}

	quick_sort_oO( val, idx, beg, l );

	// cleaning
	delete [] val;

	return mu;
}
void PScolor( Mat& img, Mat& rgb_img, Mat& out, int d, int phi, int target_distrib )
{
	// allocate tmp vars
	int l = img.rows * img.cols;
	int*   idx    = new int   [l];
	float* target = new float [256];

	float mu = ps_sort_color( img, rgb_img, d, idx, phi ); // ..
	
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

	hm_from_sorting( l, idx, target, img.cols, out );

	// imshow("im", out );
	// waitKey(-1);

	// cleaning
	delete [] idx;
	delete [] target;
}




////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void color_enhance00( cv::Mat& rgb_org, cv::Mat& rgb_out )
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

	// psi1 -> d=10-20 
	PS( org, out, 20, 2, 0 ); /// ....psps

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
/*
int main( int argc, char* argv[] )
{
	if( argc!=3 )
	{
		cerr<< "Error"<< endl
		    << "Usage : "<< argv[0]<< " input_filename output_filename"<< endl;
		return 0;
	}
	char* org_filename = argv[1];
	char* out_filename = argv[2];
	

	Mat org = imread( org_filename, CV_LOAD_IMAGE_COLOR );
	cerr<< org.cols<< " "<< org.rows<< endl;
	
	Mat out( org.rows, org.cols, CV_8UC3 );

	color_enhance00( org, out );

	imwrite( out_filename, out );


	return 0;
}
*/





