
#include <iostream>
using namespace std;

#include <opencv2/opencv.hpp>
using namespace cv;


double _lidem_mem_counter;

void lidem_reset_mem_counter() { _lidem_mem_counter = 0; }



///////////////////////////////////////////////////////////////////////////
void compute_local_post_gauss( Mat& img,
			       int K, unsigned char* mu, Mat* sd, Mat* prior, 
			       Mat* post )
{
	// allocate + init tmp vars
	float* proba = new float [K];

	_lidem_mem_counter += K/1048576.0;
	cout<< "LIDEM MEM "<< _lidem_mem_counter<< endl;

	for( int x = 0; x < img.cols; x++ ) 
	{
		for( int y = 0; y < img.rows; y++ ) 
		{
			unsigned char g = img.at<uchar>(y,x);
			/*
			float s = 0.0;
			for( int k = 0; k < K; k++ )
			{
				float sigma = sd[k].at<uchar>(y,x) / 255.0;
				float d = (float)(g - mu[k]) / (255.0 * sigma);
				float likelihood_k = exp(-d*d) / (sqrt(2)*sigma);
				float prior_k      = (float)prior[k].at<uchar>(y,x)/255.0;

				proba[k] = prior_k * likelihood_k;
			
				s += (0.6 * proba[k]);
				proba[k] *= 0.4;
			}
			s = (float)s/K;

			float sum = 0.0;
			for( int k = 0; k < K; k++ )
			{
				proba[k] += s;
				sum += proba[k];
			}
			*/
			
			float sum = 0.0;
			for( int k = 0; k < K; k++ )
			{
				float sigma = sd[k].at<uchar>(y,x) / 255.0;
				float d = (float)(g - mu[k]) / (255.0 * sigma);
				float likelihood_k = exp(-d*d) / (sqrt(2)*sigma);
				float prior_k      = (float)prior[k].at<uchar>(y,x)/255.0;

				proba[k] = prior_k * likelihood_k;
				sum += proba[k];
			}
			
			for( int k = 0; k < K; k++ )
			{
				post[k].at<uchar>(y,x) = saturate_cast<uchar>( 255*(float)proba[k]/sum );
			}
		}
	}	
/*
	// Blur
	for( int k = 0; k < K; k++ )
	{
		GaussianBlur( post[k], post[k], Size(11,11), 0, 0 );
	}
*/

	_lidem_mem_counter -= K/1048576.0;
	cout<< "LIDEM MEM "<< _lidem_mem_counter<< endl;

	// cleaning
	delete [] proba;
}


void compute_local_post_laplace( Mat& img,
				 int K, unsigned char* mu, Mat* sd, Mat* prior, 
				 Mat* post )
{
	// allocate tmp vars
	float* proba = new float [K];

	_lidem_mem_counter += K/1048576.0;
	cout<< "LIDEM MEM "<< _lidem_mem_counter<< endl;

	for( int x = 0; x < img.cols; x++ ) 
	{
		for( int y = 0; y < img.rows; y++ ) 
		{
			unsigned char g = img.at<uchar>(y,x);

			/*
			float s = 0.0;
			for( int k = 0; k < K; k++ )
			{
				float sigma = sd[k].at<uchar>(y,x) / 255.0;
				float b = sqrt(2) * sigma;
				
				float d = (float)(g - mu[k]) / (255.0 * b);
				float likelihood_k = exp(-fabs(d)) / (2*b);


				float prior_k      = (float)prior[k].at<uchar>(y,x)/255.0;

				proba[k] = prior_k * likelihood_k;

				s += (0.01 * proba[k]);
				proba[k] *= 0.99;
			}
			s = (float)s/K;

			float sum = 0.0;
			for( int k = 0; k < K; k++ )
			{
				proba[k] += s;
				sum += proba[k];
			}
			*/

			float sum = 0.0;
			for( int k = 0; k < K; k++ )
			{
				float sigma = sd[k].at<uchar>(y,x) / 255.0;
				float b = sqrt(2) * sigma;
				
				float d = (float)(g - mu[k]) / (255.0 * b);
				float likelihood_k = exp(-fabs(d)) / (2*b);


				float prior_k      = (float)prior[k].at<uchar>(y,x)/255.0;

				proba[k] = prior_k * likelihood_k;
				sum += proba[k];
			}

			for( int k = 0; k < K; k++ )
			{
				post[k].at<uchar>(y,x) = saturate_cast<uchar>( 255*(float)proba[k]/sum );
			}
		}
	}	

	_lidem_mem_counter -= K/1048576.0;
	cout<< "LIDEM MEM "<< _lidem_mem_counter<< endl;

	// cleaning
	delete [] proba;
}

///////////////////////////////////////////////////////////////////////////
int compute_mean( Mat& img, Mat& post )
{
	float m = 0.0, s = 0.0;
	for( int x = 0; x < img.cols; x++ ) 
	{
		for( int y = 0; y < img.rows; y++ ) 
		{
			float p = (float)post.at<uchar>(y,x) / 255.0;
			float g = (float)img.at<uchar>(y,x) / 255.0;

			m += p*g;
			s += p;
		}
	}
	
	return (s < 10 ? -1 : 255 * (float)m/s);
}

///////////////////////////////////////////////////////////////////////////

void compute_local_sd( Mat& img, int d, unsigned char mu, Mat& post, Mat& sd, float sigma_min )
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
	_lidem_mem_counter += (4.0 * ff);
	cout<< "LIDEM MEM>> "<< _lidem_mem_counter<< endl;

	// compute integral images
	for( int x = 0; x < w; x++ ) 
	{
		for( int y = 0; y < h; y++ ) 
		{
			float p = (float)post.at<uchar>(y,x) / 255.0;

			int   g = img.at<unsigned char>(y,x);
			float d = (float)(g - mu)/255.0;
			d *= d;
			
			s[x][y]  = (y==0 ? 0 : s[x][y-1])  + d*p;
			ii[x][y] = (x==0 ? 0 : ii[x-1][y]) + s[x][y];

			s2[x][y]  = (y==0 ? 0 : s2[x][y-1]) + p;
			ii2[x][y] = (x==0 ? 0 : ii2[x-1][y]) + s2[x][y];
		}
	}


	// do processing
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

			// ...
			float u = ii[x1][y1] + ii[x0][y0] - ii[x1][y0] - ii[x0][y1];
			float v = ii2[x1][y1] + ii2[x0][y0] - ii2[x1][y0] - ii2[x0][y1];

			float sigma = sqrt((float)u/v);
			
			sd.at<uchar>(y,x) = saturate_cast<uchar>(255 * sigma);
			if( sd.at<uchar>(y,x) < sigma_min )
			{
				sd.at<uchar>(y,x) = sigma_min; // ...
			}
		}
	}

	_lidem_mem_counter -= (4.0 * ff);
	cout<< "LIDEM MEM<< "<< _lidem_mem_counter<< endl;


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

///////////////////////////////////////////////////////////////////////////

void compute_prior( Mat& img, int d, Mat& post, Mat& prior )
{
	// allocate tmp vars
	int w = post.cols;
	int h = post.rows;
	float **ii  = new float* [w];
	float **s   = new float* [w];
	for( int x = 0; x < w; x++ ) 
	{
		ii[x]  = new float [h];
		s[x]   = new float [h];
	}

	double  ff = (w * h)/1048576.0;
	_lidem_mem_counter += (2.0 * ff);
	cout<< "LIDEM MEM>> "<< _lidem_mem_counter<< endl;

	// compute integral images
	for( int x = 0; x < w; x++ ) 
	{
		for( int y = 0; y < h; y++ ) 
		{
			float d = (float)post.at<unsigned char>(y,x) /255.0;
			
			s[x][y]  = (y==0 ? 0 : s[x][y-1])  + d;
			ii[x][y] = (x==0 ? 0 : ii[x-1][y]) + s[x][y];
		}
	}


	// do processing
	int d2 = d/2; 
	for( int x = 0; x < w; x++ ) 
	{
		for( int y = 0; y < h; y++ ) 
		{
			// get bounding box
			int x0 = x - d2, x1 = x + d2;
			int y0 = y - d2, y1 = y + d2;
			if( x0 <  0 ) { x0 = 0;   }
			if( x1 >= w ) { x1 = w-1; }
			if( y0 <  0 ) { y0 = 0;   }
			if( y1 >= h ) { y1 = h-1; }

			float dd = (x1-x0)*(y1-y0);

			// compute prior
			float m = ( (x1 < w && y1 < h ? ii[x1][y1] : 0)
				    + (0 <= x0 && 0 <= y0 ? ii[x0][y0] : 0)
				    - (x1 < w  && 0 <= y0 ? ii[x1][y0] : 0)
				    - (0 <= x0 && y1 < h  ? ii[x0][y1] : 0) ) / dd;
			
			prior.at<uchar>(y,x) = saturate_cast<uchar>(255 * m);
		}
	}

	_lidem_mem_counter -= (2.0 * ff);
	cout<< "LIDEM MEM<< "<< _lidem_mem_counter<< endl;

	// cleaning
	for( int x = 0; x < w; x++ )
	{
		delete [] ii[x];
		delete [] s[x];
	}
	delete [] ii;
	delete [] s;
}

void compute_prior( Mat& img, int d, int K, Mat* post, Mat* prior )
{
	for( int k = 0; k < K; k++ )
	{
		compute_prior( img, d, post[k], prior[k] );
	}
}

///////////////////////////////////////////////////////////////////////////

void local_enhance_gauss( Mat& img, 
			  int K, unsigned char* mu, Mat* sd, Mat* prior, 
			  Mat& out )
{
	for( int x = 0; x < img.cols; x++ ) 
	{
		for( int y = 0; y < img.rows; y++ ) 
		{
			unsigned char g = img.at<uchar>(y,x);
			
			float v = 0.0, s = 0.0;
			for( int k = 0; k < K; k++ )
			{
				float p = (float)prior[k].at<uchar>(y,x) / 255.0;

				if( p > 0 )
				{
					float f = 0.5 * (1 + erf( (float)(g - mu[k])/(sqrt(2)*sd[k].at<uchar>(y,x)) ) );
					
					v += (f*p);
					s += p;
				}
			}
			
			out.at<uchar>(y,x) = saturate_cast<uchar>(255 * v/s);
		}
	}	
}

void local_enhance_laplace( Mat& img, 
			    int K, unsigned char* mu, Mat* sd, Mat* prior, 
			    Mat& out )
{
	for( int x = 0; x < img.cols; x++ ) 
	{
		for( int y = 0; y < img.rows; y++ ) 
		{
			unsigned char g = img.at<uchar>(y,x);
			
			float v = 0.0, s = 0.0;
			for( int k = 0; k < K; k++ )
			{
				float p = (float)prior[k].at<uchar>(y,x) / 255.0;

				if( p > 0 )
				{
					int sgn = (g - mu[k] > 0 ? 1 : -1);
					double d = (float)(g - mu[k])/sd[k].at<uchar>(y,x);
					
					float f = 0.5 * (1 + sgn *(1 - exp(-sqrt(2)*abs(d))));

					v += (f*p);
					s += p;
				}
			}
			
			out.at<uchar>(y,x) = saturate_cast<uchar>(255 * v/s);
		}
	}	
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
void LIDE_EM_mixture( Mat& org, int model, int K, int d, float sigma_min, 
		      unsigned char* mu,
		      Mat* sd,
		      Mat* prior )
{
	//////////////////////////////////////////////////////////////////////////////
	// allocate + init mean, sd, prior
	Mat* post  = new Mat [K];
	for( int k = 0; k < K; k++ )
	{
		mu[k] = saturate_cast<uchar>( (k+0.5)*(float)255/K );

		sd[k].create   ( org.rows, org.cols, CV_8U );
		prior[k].create( org.rows, org.cols, CV_8U );
		post[k].create ( org.rows, org.cols, CV_8U );

		for( int x = 0; x < org.cols; x++ ) 
		{
			for( int y = 0; y < org.rows; y++ ) 
			{
				sd[k].at<uchar>(y,x) = saturate_cast<uchar>((float)255.0/K);
				prior[k].at<uchar>(y,x) = saturate_cast<uchar>(255 * (float)1.0/K);
			}
		}

	}

	double  ff = (org.rows * org.cols)/1048576.0; 
	_lidem_mem_counter += (3.0 * K * ff);
	cout<< "LIDEM MEM>> "<< _lidem_mem_counter<< endl;


	// Mat out( org.rows, org.cols, CV_8UC1 );

	//////////////////////////////////////////////////////////////////////////////
	// update via EM algo	!!!
	for( int iter = 0; iter < 5; iter++ )
	{
		if( model == 0 ) { compute_local_post_gauss  ( org, K, mu, sd, prior, post ); }
		else             { compute_local_post_laplace( org, K, mu, sd, prior, post ); }

		compute_prior( org, d, K, post, prior );

		for( int k = 0; k < K; k++ )
		{
			int j = compute_mean( org, post[k] );
			if( j >= 0 )
			{
				mu[k] = saturate_cast<uchar>(j);
				
				compute_local_sd( org, d, mu[k], post[k], sd[k], sigma_min );
			}
			else
			{
				// should not arrive here
				for( int x = 0; x < org.cols; x++ ) 
				{
					for( int y = 0; y < org.rows; y++ ) 
					{
						prior[k].at<uchar>(y,x) = 0;
					}
				}				
			}
		}

		/*
		char winname[1000];
		for( int k = 0; k < K; k++ )
		{
			sprintf( winname, "Compo_%d_%d.png", k, iter );
			imwrite( winname, sd[k] );
		}
		*/
		
		/*
		int maxv=0, minv=255;
		for( int x = 0; x < org.cols; x++ ) 
		{
			for( int y = 0; y < org.rows; y++ ) 
			{
				int s = 0;
				for( int k = 0; k < K; k++ )
				{
					s += sd[k].at<uchar>(y,x);
				}

				out.at<uchar>(y,x) = saturate_cast<uchar>(s);

				if( maxv < out.at<uchar>(y,x) ) { maxv = out.at<uchar>(y,x); }
				if( minv > out.at<uchar>(y,x) ) { minv = out.at<uchar>(y,x); }
			}
		}				

		cerr<< minv<< " "<< maxv<< endl;

		char winname[1000];
		sprintf( winname, "toti_%d.png", iter );
		imwrite( winname, out );
		*/
	}

	cerr<< "MU = [ ";
	for( int k = 0; k < K; k++ )
	{
		cerr<< (int)mu[k]<< " ";
	}
	cerr<< "]";

	_lidem_mem_counter -= (3.0 * K * ff);
	cout<< "LIDEM MEM<< "<< _lidem_mem_counter<< endl;

/*
	char winname[1000];
	for( int k = 0; k < K; k++ )
	{
		// sprintf( winname, "Compo %d", k );
		// imshow( winname, prior[k] );

		sprintf( winname, "Compo_%d.png", k );
		imwrite( winname, prior[k] );
	}
	// waitKey(-1);
	*/
	//////////////////////////////////////////////////////////////////////////////
	// cleaning
	delete [] post;

}


void LIDE_mixture( Mat& org, Mat& out, int model, int K, int d, float sigma_min )
{
	//////////////////////////////////////////////////////////////////////////////
	// allocate 
	unsigned char* mu = new unsigned char [K];
	Mat* sd    = new Mat [K];
	Mat* prior = new Mat [K];

	LIDE_EM_mixture( org, model, K, d, sigma_min, mu, sd, prior );
	
	if( model == 0 ) { local_enhance_gauss  ( org, K, mu, sd, prior, out ); }
	else             { local_enhance_laplace( org, K, mu, sd, prior, out ); }
	
	
	//////////////////////////////////////////////////////////////////////////////
	// cleaning
	delete [] mu;
	delete [] sd;
	delete [] prior;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "hm.h"
void local_hm_gauss( Mat& img, 
		     int K, unsigned char* mu, Mat* sd, Mat* prior, 
		     int n, float* cdf, 
		     Mat& out )
{
	for( int x = 0; x < img.cols; x++ ) 
	{
		for( int y = 0; y < img.rows; y++ ) 
		{
			unsigned char g = img.at<uchar>(y,x);
			
			float v = 0.0, s = 0.0;
			for( int k = 0; k < K; k++ )
			{
				float p = (float)prior[k].at<uchar>(y,x) / 255.0;

				if( p > 0 )
				{
					float f = 0.5 * (1 + erf( (float)(g - mu[k])/(sqrt(2)*sd[k].at<uchar>(y,x)) ) );
					
					v += (f*p);
					s += p;
				}
			}
			v = (float)v/s;
			
			int newg = inverse_cdf( n, cdf, v );

			out.at<uchar>(y,x) = saturate_cast<uchar>(newg);
		}
	}	
}

void local_hm_laplace( Mat& img, 
		       int K, unsigned char* mu, Mat* sd, Mat* prior, 
		       int n, float* cdf, 
		       Mat& out )
{
	for( int x = 0; x < img.cols; x++ ) 
	{
		for( int y = 0; y < img.rows; y++ ) 
		{
			unsigned char g = img.at<uchar>(y,x);
			
			float v = 0.0, s = 0.0;
			for( int k = 0; k < K; k++ )
			{
				float p = (float)prior[k].at<uchar>(y,x) / 255.0;

				if( p > 0 )
				{
					int sgn = (g - mu[k] > 0 ? 1 : -1);
					double d = (float)(g - mu[k])/sd[k].at<uchar>(y,x);
					
					float f = 0.5 * (1 + sgn *(1 - exp(-sqrt(2)*abs(d))));

					v += (f*p);
					s += p;
				}
			}
			v = (float)v/s;

			int newg = inverse_cdf( n, cdf, v );
			
			out.at<uchar>(y,x) = saturate_cast<uchar>(newg);
		}
	}	
}


void LIDEHM_mixture( Mat& org, Mat& out, int model, int K, int d, float sigma_min, int n, float* hist )
{
	//////////////////////////////////////////////////////////////////////////////
	// allocate 
	float* cdf = new float [n];
	unsigned char* mu = new unsigned char [K];
	Mat* sd    = new Mat [K];
	Mat* prior = new Mat [K];

	LIDE_EM_mixture( org, model, K, d, sigma_min, mu, sd, prior );
	
	build_cdf( n, hist, cdf );

	if( model == 0 ) { local_hm_gauss  ( org, K, mu, sd, prior, n, cdf, out ); }
	else             { local_hm_laplace( org, K, mu, sd, prior, n, cdf, out ); }
	
	
	//////////////////////////////////////////////////////////////////////////////
	// cleaning
	delete [] cdf;
	delete [] mu;
	delete [] sd;
	delete [] prior;
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "bpheme.h"
#include "fhsabp.h"
#include "he.h"

void LIDE_mixture( Mat& img, Mat& out, int model, int K, int d, float sigma_min, int target_distrib )
{
	if( target_distrib == 0 ) // normal
	{
		LIDE_mixture( img, out, model, K, d, sigma_min );
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
		case 1 : build_target_distrib_bpheme( mu, 256, target ); cerr<< "BPHEME"<< endl; break;
		case 2 : build_target_distrib_fhsabp( mu, target );      cerr<< "FHSABP"<< endl; break;
		default : // method 0 
			for( int i = 0; i < 256; i++ )
			{
				target[i] = (float)1.0/256.0;
			}
		}

		LIDEHM_mixture( img, out, model, K, d, sigma_min, 256, target );

		delete [] target;
	}
}



/*

		{
			Mat oo (258, 258, CV_8UC3);
	
			float* hist = new float [256];
			compute_hist( out, hist );

			float maxv = 0;
			for( int i = 0; i < 256; i++ )
			{
				if( maxv < hist[i] ) 
				{
					maxv = hist[i];
				}
			}
			maxv = (float)7.0/256.0;

			rectangle( oo, Point(0,0), Point(258,258), CV_RGB(255,255,255), CV_FILLED);

			for( int i = 0; i < 256; i++ )
			{
				int y = 1 + (int)(256*(float)hist[i]/maxv);
		
				line( oo, Point(1+i, 257), Point(1+i, 257-y), CV_RGB(0,0,0) );
			}
	
			imwrite( "toto00.png", oo );

			// exit(0);
		}


*/
