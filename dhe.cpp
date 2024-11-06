


#include <cmath>
#include <iostream>
using namespace std;

#include "he.h"
#include "dhe.h"
using namespace cv;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool interval_is_gaussian( int beg, int end, float* hist, int& l, int& r )
{
	float mu = 0.0, sum = 0.0;;
	for( int i = beg; i < end; i++ )
	{
		mu  += i*hist[i];
		sum +=   hist[i];
	}
	mu = (float)mu/sum;

	float sd = 0.0;
	for( int i = beg; i < end; i++ )
	{
		sd += ((i - mu) * (i - mu)  * hist[i]);
	}
	sd = sqrt( (float)sd/sum );

	l = (int)(mu-sd);
	r = (int)(mu+sd);
	if( l< beg ) { l = beg;   }
	if( r>=end ) { r = end-1; }
	float s = 0.0;
	for( int i = l; i < r; i++ )
	{
		s += hist[i];
	}

	return ((float)s/sum > 0.683); // 
}

void smooth_hist( int L, float* hist )
{
	float oldv = hist[0];
	for( int i = 1; i < L-1; i++ )
	{
		float v = (oldv + 2*hist[i] + hist[i+1]) / 4.0;
		oldv    = hist[i];
		hist[i] = v;
	}
}

void partition_from_minima( int L, float* hist, int& m, int* thresholds )
{
	m = 0;
	for( int i = 1; i < L-1; i++ )
	{
		if( hist[i-1]>=hist[i] && hist[i]<hist[i+1] )
		{
			thresholds[m] = i;
			m++;
		}
	}
	if( thresholds[m-1] == L-2 )
	{
		thresholds[m-1] = L-1;
	}
	else if( m<L )
	{
		thresholds[m] = L-1;
		m++;
	}
}


#include "sort.h"
void dynamic_histogram_partition( int L, float* hist, int& m, int* thresholds, int mind )
{
	// allocate tmp vars
	float* hist_tmp = new float [L];
	for( int i = 0; i < L; i++ )
	{
		hist_tmp[i] = hist[i];
	}


	// step 1) partition using minima
	bool ok = false;
	for( int iter = 0; iter < 200 && !ok; iter++ )
	{
		smooth_hist( L, hist_tmp );

		partition_from_minima( L, hist_tmp, m, thresholds );

		ok = (m>=2);
		for( int i = 1; i < m && ok; i++ )
		{
			ok = (abs(thresholds[i] - thresholds[i-1]) > mind);
		}
	}


	// step 2) check each portion, if not Gaussian split it into 3 intervals
	int mm = m-1, l = 0, r = 0;
	for( int i = 0; i < mm; i++ )
	{
		if( !interval_is_gaussian( thresholds[i], thresholds[i+1], hist, l, r ) )
		{
			if( l-thresholds[i]   > mind ) { thresholds[m] = l; m++; }
			if( thresholds[i+1]-r > mind ) { thresholds[m] = r; m++; }
		}
	}

	// sort the thresholds
	quick_sort_oO( thresholds, 0, m );

	for( int i = 0; i < m; i++ )
	{
		cerr<< i<< " "<< thresholds[i]<< endl;
	}

	// cleaning
	delete [] hist_tmp;
}


void DHE( Mat& img, Mat& out, int min_length )
{
	// allocate tmp vars
	int L = 256;
	float* hist       = new float [L];
	int*   thresholds = new int   [L];
	int*   newval     = new int   [L];


	compute_hist( img, hist );

	int m = 0;
	dynamic_histogram_partition( L, hist, m, thresholds, min_length );

	equalize_hist_interval( L, hist, m, thresholds, newval );

	grayscale_remap( img, newval, out );

	// cleaning
	delete [] hist;
	delete [] thresholds;
	delete [] newval;
}
