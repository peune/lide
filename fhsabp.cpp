



#include <cmath>
#include <iostream>
using namespace std;

#include "he.h"
#include "bpheme.h"
#include "fhsabp.h"
using namespace cv;


void compute_target_low( float mu, float* hist )
{
	float x0 = floor(3*mu) + 1;
	float a  = (float) (-6*x0 + 12*mu)  / (x0*(x0+1)*(x0+2));
	float b  = (float) (4*x0 - 6*mu +2) / ((x0+1)*(x0+2));

	float s = 0.0;
	for( int i = 0; i < 256; i++ )
	{
		hist[i] = a*i + b;
		if( hist[i] < 0 ) { hist[i] = 0; }

		s += hist[i];
	}
	for( int i = 0; i < 256; i++ )
	{
		hist[i] = (float)hist[i]/s;
	}
}

void compute_target_mid( float mu, float* hist )
{
	float a = (float)(mu - 127.5) / 1398080.0;
	float b = (float)(511 - 3*mu) / 32896.0;
	
	float s = 0.0;
	for( int i = 0; i < 256; i++ )
	{
		hist[i] = a*i + b;

		s += hist[i];
	}
	for( int i = 0; i < 256; i++ )
	{
		hist[i] = (float)hist[i]/s;
	}
}

void compute_target_high( float mu, float* hist )
{
	// allocate tmp vars
	float* p = new float [256];

	compute_target_low( 255-mu, p );

	for( int i = 0; i < 256; i++ )
	{
		hist[i] = p[255 - i];
	}

	// cleaning
	delete [] p;
}

void build_target_distrib_fhsabp( float mu, float* target )
{
	if( mu < 84.67 )
	{
		cerr<< "low "<< mu<< endl;
		compute_target_low( mu, target );
	}
	else if( mu < 170.33 )
	{
		cerr<< "mid "<< mu<< endl;
		compute_target_mid( mu, target );
	}
	else
	{
		cerr<< "high "<< mu<< endl;
		compute_target_high( mu, target );
	}

/*
	for( int i = 0; i < 256; i++ )
	{
		cout<< i<< " "<< f[i]<< endl;
	}
	exit(0);
*/
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FHSABP( Mat& img, Mat& out )
{
	// allocate tmp vars
	float* hist   = new float [256];
	float* target = new float [256];
	int*   newval = new int   [256];

	float mu = compute_hist( img, hist );

	build_target_distrib_fhsabp( mu, target );

	histogram_matching( 256, hist, target, newval );

	float mm = grayscale_remap( img, newval, out );
	// cerr<< "mm = "<< mm<< endl;

	// cleaning
	delete [] hist;
	delete [] target;
	delete [] newval;
}


