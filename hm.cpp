


#include <cmath>
#include <iostream>
using namespace std;

#include "he.h"
#include "hm.h"
using namespace cv;


//////////////////////////////////////////////////////////////////////////////////////////////

void histogram_matching( int nbin, float* h, float* f, int* newval )
{
	// allocate tmp vars
	float* F = new float [nbin];
	float* H = new float [nbin];
	
	for( int i = 0; i < nbin; i++ )
	{
		F[i] = f[i] + (i==0 ? 0 : F[i-1]);
		H[i] = h[i] + (i==0 ? 0 : H[i-1]);
	}

	for( int i = 0; i < nbin; i++ )
	{
		int j = (i==0 ? 0 : newval[i-1]);
		while( j<nbin && F[j]<=H[i] ) { j++; }

		newval[i] = (j == 0 ? 0 : j-1);
	}

	// cleaning
	delete [] F;
	delete [] H;
}


//////////////////////////////////////////////////////////////////////////////////////////////

void build_cdf( int n, float* hist, float* cdf )
{
	for( int i = 0; i < n; i++ )
	{
		cdf[i] = hist[i] + (i== 0 ? 0 : cdf[i-1]);
	}
}

int inverse_cdf( int n, float* cdf, float v )
{
	int j = 0;
	while( j<n && cdf[j]<=v ) { j++; }

	return j;
}

//////////////////////////////////////////////////////////////////////////////////////////////


