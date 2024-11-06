


#include <cmath>
#include <iostream>
using namespace std;

#include "he.h"
using namespace cv;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
  Followed K. Singh and R. Kapoor
  "Image enhancement using Exposure based Sub Image Histogram Equalization"
  in Pattern Recognition Letters 36 (2014) 10-14

  Clipping:
  Tc = (1/L) sum_k h(k)
  If h(k)>=Tc then h(k) = Tc

  Exposure:
  exposure = (1/L) (sum_k h(k) k) / (sum_k h(k))
  h = histogram
  L = grayscale level

  Xa = L(1-exposure)
  Then perform equalization for 2 parts: <Xa and >Xa
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ESIHE( Mat & img, Mat &out )
{
	// allocate tmp vars
	int L = 256;
	float* hist       = new float [L];
	int*   newval     = new int   [L];
	int*   thresholds = new int   [1]; // ...

	compute_hist( img, hist );

	// find exposure and clipping threshold
	float Tc = 0.0, exposure = 0.0;
	for( int k = 0; k < L; k++ )
	{
		Tc += hist[k];
		exposure += (hist[k] * k);
	}
	exposure = (float)exposure / (L * Tc);
	Tc = (float)Tc/L;

	// clip the histogram
	for( int k = 0; k < L; k++ )
	{
		if( hist[k] >= Tc ) { hist[k] = Tc; }
	}
	
	// set up the threshold
	thresholds[0] = L * (1-exposure);

	// perform equalization
	equalize_hist_interval( L, hist, 1, thresholds, newval );

	grayscale_remap( img, newval, out );

	// cleaning
	delete [] hist;
	delete [] newval;
	delete [] thresholds;
}



