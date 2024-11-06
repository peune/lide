



#include <cmath>
#include <iostream>
using namespace std;

#include "he.h"
#include "mhe.h"
using namespace cv;

////////////////////////////////////////////////////////////////////////////////////////////////////

/*
  Followed M. Luessi, M. Eichmann, G.M. Schuster, and Aggelos K. Katsaggelos
  "New results on efficient optimal multilevel image thresholding"
  in IEEE International Conference on Image Processing, 2006
  pp. 773-776


  Objective: 
  find thresholds t_1,...,t_{M-1} maximizing J(t_1,...,t_{M-1}; L)

  Hyp: 
  J(t_1,...,t_{m-1}; l) = sum_{k=1,..,m} loss(t_{k-1}, t_k] with 0<=t_1<=...<=t_{m-1}=l

  Dynamic programming recursion:
  J*(m,l) = max sum_{k=1,..,m} loss(t_{k-1}, t_k] 
  J*(m,l) = max { sum_{k=1,..,m-1} loss(t_{k-1}, t_k]  + loss(t_{m-1},l] }
  J*(m,l) = max { J*(m-1,t_{m-1}) + loss(t_{m-1},l] }

  -> Noted that 'loss' function does not depend on index of segment

  Implementation: t and m start from 0
  J[0][t] = loss([0, t]), t=0,...,L-1
  P[0][t] = 0
  for m = 1 to M do
      for t = m to L do
          l* = arg max_{l<t} J[m-1][l] + loss((l, t])
          J[m][t] = J[m-1][l*] + loss(*l, t)
	  P[m][t] = l*

 */
// loss[a][b] = loss([a,b])
float compute_thresholds( int L, float** loss, int M, int* thresholds )
{
	// allocate tmp vars
	float** J = new float* [M];
	int**   P = new int*   [M];
	for( int m = 0; m < M; m++ )
	{
		J[m] = new float [L];
		P[m] = new int   [L];
	}
	
	// init
	for( int t = 0; t < L; t++ )
	{
		J[0][t] = loss[0][t];
		P[0][t] = 0;
	}

	// recursion
	for( int m = 1; m < M; m++ )
	{
		for( int t = m; t < L; t++ )
		{
			float best_v = HUGE_VAL;
			int   best_l = -1;
			for( int l = m-1; l < t; l++ )
			{
				float v = J[m-1][l] + loss[l+1][t]; 
				if( best_v > v )
				{
					best_v = v;
					best_l = l;
				}
			}
			J[m][t] = J[m-1][best_l] + loss[best_l+1][t];
			P[m][t] = best_l;
		}
	}
	
	// backtrack
	for( int m=M-1, t=L-1; m>=0 && t>=0; t=P[m][t], m-- )
	{
		if( m < 0 ) { cerr<< "OCOCOCCO  "<< m<< " "<< t<< endl; exit(0); }

		thresholds[m] = t;
	}
	

	float discrepancy = J[M-1][L-1];


	// cleaning
	for( int m = 0; m < M; m++ )
	{
		delete [] J[m];
		delete [] P[m];
	}
	delete [] J;
	delete [] P;


	return discrepancy;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////
void compute_loss_mean( int L, float* hist, float** loss )
{
	for( int i = 0; i < L; i++ )
	{
		for( int j = i; j < L; j++ )
		{
			float mean = 0.0;
			for( int t = i; t <= j; t++ )
			{
				mean += t;
			}
			mean = (float)mean/(j-i+1);

			float d = 0.0;
			for( int t = i; t <= j; t++ )
			{
				d += ((t-mean) * (t-mean) * hist[t]);
			}
			loss[i][j] = d;
			loss[j][i] = d;
		}
	}
}

void compute_loss_middle( int L, float* hist, float** loss )
{
	for( int i = 0; i < L; i++ )
	{
		for( int j = i; j < L; j++ )
		{
			float mid = (float)(i+j)/2.0;
			float d = 0.0;
			for( int t = i; t <= j; t++ )
			{
				d += ((t-mid) * (t-mid) * hist[t]);
			}
			loss[i][j] = d;
			loss[j][i] = d;
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
int mhe_partition( int L, float* hist, int* thresholds, int Mmin, int Mmax, float rho )
{
	// allocate tmp vars
	float** loss = new float* [L];
	for( int i = 0; i < L; i++ )
	{
		loss[i] = new float [L];
	}


	compute_loss_middle( L, hist, loss );


	// find optimal number of segment M
	float best_score = HUGE_VAL;
	int   best_M     = -1;
	for( int M = Mmin; M <= Mmax; M++ )
	{
		float disc = compute_thresholds( L, loss, M, thresholds );
		float c = log(M) / log(2.0);

		float score = rho*sqrt(disc) + c*c;

		if( best_score > score )
		{
			best_score = score;
			best_M     = M;
		}
	}

	// and use best_M
	compute_thresholds( L, loss, best_M, thresholds );

	// cleaning
	for( int i = 0; i < L; i++ )
	{
		delete [] loss[i];
	}
	delete [] loss;

	return best_M;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
void MHE( Mat& img, Mat& out, int Mmin, int Mmax, float rho )
{
	// allocate tmp vars
	int L = 256;
	float* hist       = new float [L];
	int*   thresholds = new int   [L];
	int*   newval     = new int   [L];


	compute_hist( img, hist );

	int best_M = mhe_partition( L, hist, thresholds, Mmin, Mmax, rho );

	equalize_hist_interval( L, hist, best_M, thresholds, newval );

	grayscale_remap( img, newval, out );


	// cleaning
	delete [] hist;
	delete [] thresholds;
	delete [] newval;
}



/*
void MHE( Mat& img, Mat& out, int Mmin, int Mmax, float rho )
{
	// allocate tmp vars
	int L = 256;
	float* hist       = new float [L];
	int*   thresholds = new int   [L];
	int*   newval     = new int   [L];
	float** loss = new float* [L];
	for( int i = 0; i < L; i++ )
	{
		loss[i] = new float [L];
	}


	compute_hist( img, hist );

	compute_loss_middle( L, hist, loss );


	// find optimal number of segment M
	float best_score = HUGE_VAL;
	int   best_M     = -1;
	for( int M = Mmin; M <= Mmax; M++ )
	{
		float disc = compute_thresholds( L, loss, M, thresholds );
		float c = log(M) / log(2.0);

		float score = rho*sqrt(disc) + c*c;

		if( best_score > score )
		{
			best_score = score;
			best_M     = M;
		}
	}

	cerr<< "Best M = "<< best_M<< endl;

	// and use best_M
	compute_thresholds( L, loss, best_M, thresholds );
	
	equalize_hist_interval( L, hist, best_M, thresholds, newval );

	grayscale_remap( img, newval, out );


	// cleaning
	for( int i = 0; i < L; i++ )
	{
		delete [] loss[i];
	}
	delete [] loss;
	delete [] hist;
	delete [] thresholds;
	delete [] newval;
}

*/


