
/*
  Histogram Equalization using Gaussian Mixture Model
*/

#include <cmath>
#include <iostream>
using namespace std;

#include <opencv2/opencv.hpp>
using namespace cv;

#include "hegmm.h"


float gauss_pdf( float mu, float sigma, float x )
{
	float d = (float)(mu-x)/sigma; // ...
	float c = (float)1.0 / (sqrt( 2 * M_PI ) * sigma);

	return c * exp(-0.5*d*d);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////
// EM training

void compute_grayscale_posterior_proba( float* weight, // for each grayscale level
					int K, float* mu, float* sigma, float* pi, 
					float** post )
{
	for( int i = 0; i < 256; i++ )
	{
		if( weight[i]>1e-5 )
		{
			float s = 0.0;
			int maxk = 0;
			for( int k = 0; k < K; k++ )
			{
				if( pi[k] < 1e-5 )
				{
					post[i][k] = 0.0;
				}
				else
				{
					post[i][k] = (pi[k] * gauss_pdf(mu[k], sigma[k], i));
				}
				s += post[i][k];

				if( post[i][maxk] < post[i][k] ) 
				{
					maxk = k;
				}
			}
			for( int k = 0; k < K; k++ )
			{
				post[i][k] = (s < 1e-8 ?
					      (k == maxk ? 1 : 0) :
					      (float)post[i][k]/s);
			}
		}
	}
}

void update_grayscale_component_mu_sigma( float*weight,
					  int K, float* mu, float* sigma, float* pi, 
					  float** post, int k0 )
{
	float f = 0.0;
	for( int k = 0; k < K; k++ )
	{
		float z = 0.0, m = 0.0, s = 0.0;
		for( int i = 0; i < 256; i++ )
		{
			if( weight[i]>1e-5 )
			{
				m += (weight[i] * post[i][k] * i);
				s += (weight[i] * post[i][k] * pow( (double)(i-mu[k]), 2.0 ));
				z += (weight[i] * post[i][k]);
			}
		}
	
		if( k == k0 && z > 1e-5 )
		{
			mu[k]    = (float)m/z;
			sigma[k] = sqrt( (float)s/z );
			if( sigma[k] < 5 ) { sigma[k] = 5; } // ...
		}

		pi[k] = z;
		f += pi[k];
	}
}

void gmm_grayscale_component_em_training( float* weight, float xmin, float xmax,
					  int K, float* mu, float* sigma, float* pi ) 
{
	// allocate tmp vars
	float** post = new float* [256];
	for( int i = 0; i < 256; i++ )
	{
		post[i] = new float [K];
	}

	// init
	for( int k = 0; k < K; k++ )
	{
		// mu[k]    = 256 * (float)(k+0.5)/K;
		mu[k] = xmin + (xmax-xmin)*(float)k/K;
		
		// sigma[k] = 127.0;
		sigma[k] = (xmax-xmin)/K;

		pi[k]    = (float)1.0/K;
	}

	// iterative update
	for( int iter = 0; iter < 50; iter++ )
	{
		for( int k = 0; k < K; k++ )
		{
			if( pi[k]>1e-5 )
			{
				compute_grayscale_posterior_proba( weight, K, mu, sigma, pi, post );
				
				update_grayscale_component_mu_sigma( weight, K, mu, sigma, pi, post, k );
			}
		}
		if( iter%5==0 ) { cerr<< "_"; }
	}
	cerr<< endl;

	// cleaning
	for( int i = 0; i < 256; i++ )
	{
		delete [] post[i];
	}
	delete [] post;
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////






void compute_posterior_proba( int n, uchar* x,
			      int K, float* mu, float* sigma, float* pi, 
			      float** post )
{
	// allocate = init tmp var
	float** gval = new float* [K];
	for( int k = 0; k < K; k++ )
	{
		gval[k] = new float [256];
		for( int g = 0; g < 256; g++ )
		{
			gval[k][g] = gauss_pdf(mu[k], sigma[k], g);
		}
	}

	for( int i = 0; i < n; i++ )
	{
		float s = 0.0;
		int maxk = 0;
		for( int k = 0; k < K; k++ )
		{
			if( pi[k] < 1e-5 )
			{
				post[i][k] = 0.0;
			}
			else
			{
				// post[i][k] = (pi[k] * gauss_pdf(mu[k], sigma[k], x[i]));
				post[i][k] = (pi[k] * gval[k][x[i]]);
			}
			s += post[i][k];

			if( post[i][maxk] < post[i][k] ) 
			{
				maxk = k;
			}
		}
		for( int k = 0; k < K; k++ )
		{
			post[i][k] = (s < 1e-8 ?
				      (k == maxk ? 1 : 0) :
				      (float)post[i][k]/s);
		}
	}


	// cleaning
	for( int k = 0; k < K; k++ )
	{
		delete [] gval[k];
	}
	delete [] gval;
}

void update_mu_sigma( int K, float* mu, float* sigma, float* pi, 
		      int n, uchar* x, float** post )
{
	for( int k = 0; k < K; k++ )
	{
		float z = 0.0, m = 0.0, s = 0.0;
		for( int i = 0; i < n; i++ )
		{
			m += (post[i][k] * x[i]);
			s += (post[i][k] * pow( (double)(x[i]-mu[k]), 2.0 ));
			z += post[i][k];
		}

		if( z > 1e-5 )
		{
			mu[k]    = (float)m/z;
			sigma[k] = sqrt( (float)s/z );
			if( sigma[k] < 5 ) { sigma[k] = 5; } // ...
		}

		pi[k] = (float)z/n;
	}
}

void gmm_em_training( int n, uchar* x, float xmin, float xmax,
		      int K, float* mu, float* sigma, float* pi ) 
{
	// allocate tmp vars
	float** post = new float* [n];
	for( int i = 0; i < n; i++ )
	{
		post[i] = new float [K];
	}

	// init
	for( int k = 0; k < K; k++ )
	{
		// mu[k]    = 256 * (float)(k+0.5)/K;
		mu[k] = xmin + (xmax-xmin)*(float)k/K;

		// sigma[k] = 127.0;
		sigma[k] = (xmax-xmin)/K;

		pi[k]    = (float)1.0/K;
	}

	// iterative update
	for( int iter = 0; iter < 50; iter++ )
	{
		compute_posterior_proba( n, x, K, mu, sigma, pi, post );

		update_mu_sigma( K, mu, sigma, pi, n, x, post );
	}

	// cleaning
	for( int i = 0; i < n; i++ )
	{
		delete [] post[i];
	}
	delete [] post;
}



void update_component_mu_sigma( int K, float* mu, float* sigma, float* pi, 
				int n, uchar* x, float** post, int k0 )
{
	float f = 0.0;
	for( int k = 0; k < K; k++ )
	{
		float z = 0.0, m = 0.0, s = 0.0;
		for( int i = 0; i < n; i++ )
		{
			m += (post[i][k] * x[i]);
			s += (post[i][k] * pow( (double)(x[i]-mu[k]), 2.0 ));
			z += post[i][k];
		}
	
		if( k == k0 && z > 1e-5 )
		{
			mu[k]    = (float)m/z;
			sigma[k] = sqrt( (float)s/z );
			if( sigma[k] < 5 ) { sigma[k] = 5; } // ...
		}

		pi[k] = (float)z/n;
		f += pi[k];
	}
}

void gmm_component_em_training( int n, uchar* x, float xmin, float xmax,
				int K, float* mu, float* sigma, float* pi ) 
{
	// allocate tmp vars
	float** post = new float* [n];
	for( int i = 0; i < n; i++ )
	{
		post[i] = new float [K];
	}

	// init
	for( int k = 0; k < K; k++ )
	{
		// mu[k]    = 256 * (float)(k+0.5)/K;
		mu[k] = xmin + (xmax-xmin)*(float)k/K;
		
		// sigma[k] = 127.0;
		sigma[k] = (xmax-xmin)/K;

		pi[k]    = (float)1.0/K;
	}

	// iterative update
	for( int iter = 0; iter < 50; iter++ )
	{
		for( int k = 0; k < K; k++ )
		{
			if( pi[k]>1e-5 )
			{
				compute_posterior_proba( n, x, K, mu, sigma, pi, post );
				
				update_component_mu_sigma( K, mu, sigma, pi, n, x, post, k );
			}
		}
		if( iter%5==0 ) { cerr<< "_"; }
	}
	cerr<< endl;

	// cleaning
	for( int i = 0; i < n; i++ )
	{
		delete [] post[i];
	}
	delete [] post;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////
void find_intersection( float mu1, float sigma1, float pi1,
			float mu2, float sigma2, float pi2,
			float& xs1, float& xs2 )
{
	float v1 = sigma1*sigma1, v2 = sigma2*sigma2;
	float a = (v1 - v2);
	float b = 2 * (mu1*v2 - mu2*v1);
	float c = (mu2*v1 - mu1*v2) - 2*v1*v2*log( (float)(pi2*sigma1) / (pi1*sigma2) );
	float d = sqrt(b*b - 4*a*c);
	
	xs1 = (float)(-b + d)/(2*a);
	xs2 = (float)(-b - d)/(2*a);
}

void find_significant_intersection( int K, float* mu, float* sigma, float* pi,
				    int m, int n,
				    int xmin, int xmax, 
				    int& l, float* xs ) 
{
	float xl[2];
	find_intersection( mu[m], sigma[m], pi[m],
			   mu[n], sigma[n], pi[n],
			   xl[0], xl[1] );

	// cerr<< "candidate "<< xl[0]<< " "<< xl[1]<< endl;

	for( int i = 0; i < 2; i++ )
	{
		if( xmin < xl[i] && xl[i] < xmax )
		{
			float vm = (pi[m] * gauss_pdf(mu[m], sigma[m], xl[i]));
			// cerr<< "\t"<< vm<< " : ";

			bool ok = true;
			for( int k = 0; k < K && ok; k++ )
			{
				if( k!=m && k!=n )
				{
					float v = (pi[k] * gauss_pdf(mu[k], sigma[k], xl[i]));
				
					ok = (v < vm);
					// cerr<< v<< " ";
				}
			}
		
			if( ok )
			{
				xs[l] = xl[i];
				l++;

				// cerr<< "OK";
			}
			// cerr<< endl;
		}
	}
}

#include "sort.h"
void find_significant_intersection( int K, float* mu, float* sigma, float* pi,
				    int xmin, int xmax, 
				    int& l, float* xs ) 
{
	l = 2;
	xs[0] = xmin;
	xs[1] = xmax;
	for( int m = 0; m < K; m++ )
	{
		if( pi[m]>1e-5 ) // just in case
		{
			for( int n = m+1; n < K; n++ )
			{
				if( pi[n]>1e-5 )
				{
					find_significant_intersection( K, mu, sigma, pi, m, n, xmin, xmax, l, xs );
				}
			}
		}
	}
	quick_sort_oO( xs, 0, l );
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////
float gauss_cdf( float mu, float sigma, float x )
{
	float f = 0.5 * (1 + erf( (float)(x-mu)/(sigma*sqrt(2)) ));

	return f;
}

float gmm_cdf( int K, float* mu, float* sigma, float* pi, float x )
{
	float s = 0.0;
	for( int k = 0; k < K; k++ )
	{
		s += (pi[k] * gauss_cdf( mu[k], sigma[k], x ));
	}
	return s;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
int find_dominant_gaussian( int K, float* mu, float* sigma, float* pi, float xbeg, float xend )
{
	float maxv = gauss_cdf(mu[0], sigma[0], xend) - gauss_cdf(mu[0], sigma[0], xbeg);
	int   maxk = 0;
	for( int k = 1; k < K; k++ )
	{
		float v = gauss_cdf(mu[k], sigma[k], xend) - gauss_cdf(mu[k], sigma[k], xbeg);
		if( maxv < v ) 
		{
			maxv = v;
			maxk = k;
		}
	}
	return maxk;
}

void find_dominant_gaussian( int K, float* mu, float* sigma, float* pi, 
			     int l, float* xs,
			     int* dominant_gauss )
{
	for( int i = 0; i < l-1; i++ )
	{
		dominant_gauss[i] = find_dominant_gaussian( K, mu, sigma, pi, xs[i], xs[i+1] );

		// cerr<< xs[i]<< " "<< xs[i+1]<< " :: "<< dominant_gauss[i]<< " "<< mu[dominant_gauss[i]]<< endl;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
void compute_alpha( int K, float* mu, float* sigma, float* pi, 
		    int l, float* xs,
		    int* dominant_gauss,
		    float gamma,
		    float* alpha )
{
	float s1 = 0.0;
	for( int k = 0; k < K; k++ )
	{
		s1 += pow( sigma[k], gamma );
	}
	float s2 = 0.0;
	for( int i = 0; i < l-1; i++ )
	{
		float f1 = gmm_cdf( K, mu, sigma, pi, xs[i]   );
		float f2 = gmm_cdf( K, mu, sigma, pi, xs[i+1] );

		int k = dominant_gauss[i];
		alpha[i] = ((float)pow( sigma[k], gamma ) / s1) * (f2-f1);

		s2 += alpha[i];
	}
	for( int i = 0; i < l-1; i++ )
	{
		alpha[i] = (float)alpha[i]/s2;
		// cerr<< alpha[i]<< " ";
	}
	// cerr<< endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
void find_output_intervals( int l, float* alpha, int ymin, int ymax,
			    float* ys )
{
	float s = 0.0;
	for( int i = 0; i < l; i++ )
	{
		ys[i] = ymin + s*(ymax-ymin);
		s += alpha[i];

		// cerr<< ys[i]<< " ";
	}
	// cerr<< endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
void map_model( float mux, float sigmax, 
		float xbeg, float xend,
		float ybeg, float yend,
		float& muy, float& sigmay )
{
	float a = (float)(xbeg - mux)/(xend-mux);

	muy = (float)(a*yend - ybeg) / (a - 1);
	
	sigmay = sigmax * (float)(ybeg - muy) / (xbeg - mux);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "he.h"
void init_gmm( Mat& img, int K, float* mu, float* sigma, float* pi, float& xmin, float& xmax )
{
	// allocate tmp var
	int n = img.rows * img.cols;
	uchar* x = new uchar [n];
	float* hist = new float [256];
	for( int i = 0; i < 256; i++ )
	{
		hist[i] = 0;
	}

	xmin =  HUGE_VAL;
	xmax = -HUGE_VAL;
	for( int r = 0; r < img.rows; r++ )
	{
		for( int c = 0; c < img.cols; c++ )
		{
			// int i = c + r*img.cols;
			uchar g = img.at<uchar>(r,c);
			if( xmin > g ) { xmin = g; }
			if( xmax < g ) { xmax = g; }

			hist[g]++;
		}
	}
	for( int i = 0; i < 256; i++ )
	{
		hist[i] = (float)hist[i]/(img.rows*img.cols);
	}

	gmm_grayscale_component_em_training( hist, xmin, xmax, K, mu, sigma, pi );


	// cleaning
	delete [] x;
	delete [] hist;
}
	       


////////////////////////////////////////////////////////////////////////////////////////////////////////////

void HEGMM( Mat& org, Mat& out, int K )
{
	// allocate tmp vars
	float* mu    = new float [K];
	float* sigma = new float [K];
	float* pi    = new float [K];


	float xmin=0, xmax=0;
	init_gmm( org, K, mu, sigma, pi, xmin, xmax );
	// cerr<< "GMM found"<< endl;

	int l = 0;
	float* xs = new float [K*(K-1)]; // <<======= allocate more var
	
	find_significant_intersection( K, mu, sigma, pi, xmin, xmax, l, xs );

// 	cerr<< l<< endl;

	// <<======= allocate more var
	int* dominant_gauss = new int   [l];
	float* alpha        = new float [l];
	float* ys           = new float [l];

	find_dominant_gaussian( K, mu, sigma, pi, l, xs, dominant_gauss );
	
	compute_alpha( K, mu, sigma, pi, l, xs, dominant_gauss, 0.5, alpha );

	find_output_intervals( l, alpha, 0, 255, ys );
	
	int graymap[256];
	for( int i = 0; i < l-1; i++ )
	{
		int k = dominant_gauss[i];
		float muy=0, sigmay=0;
		map_model( mu[k], sigma[k], xs[i], xs[i+1], ys[i], ys[i+1], muy, sigmay ); 

		for( int j = xs[i]; j < xs[i+1]; j++ )
		{
			int y = muy + (j-mu[k])*(float)sigmay/sigma[k];
			graymap[j] = y;
		}
        }
	for( int j = xs[l-1]; j < 256; j++ )
	{
		graymap[j] = graymap[ (int)xs[l-1]-1 ];
	}

	for( int x = 0; x < out.cols; x++ )
	{
		for( int y = 0; y < out.rows; y++ )
		{
			out.at<uchar>(y,x) = graymap[ org.at<uchar>(y,x) ];
		}
	}

	// cleaning
	delete [] mu;
	delete [] sigma;
        delete [] pi;
        delete [] xs;
        delete [] dominant_gauss;
        delete [] alpha;
        delete [] ys;
	
}







