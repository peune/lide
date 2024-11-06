
#ifndef _HM_H_
#define _HM_H_

void build_cdf( int n, float* hist, float* cdf );

int inverse_cdf( int n, float* cdf, float v );

// newval[s] = F^-1 H(s)
// h=input, f=target output
void histogram_matching( int nbin, float* h, float* f, int* newval );

#endif // _HM_H_


