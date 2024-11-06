
#ifndef _BPHEME_H_
#define _BPHEME_H_

#include <opencv2/opencv.hpp>

/*
  BPHEME
  C. Wang and Z. Ye, “Brightness preserving histogram equalization with
  maximum entropy: A variational perspective,” IEEE Trans. Consum.
  Electron., vol. 51, no. 4, pp. 1326–1334, Nov. 2005.

  Idea:
  - build target distribution f by solving
    max_f -sum_s f(s) log f(s)
    subject to f(s)>=0, for all s, sum_s f(s) = 1, sum_s s f(s) = mu

  - solution 
    f(s) = 1 if mu = 0.5 otherwise f(s) = lambda exp(lambda s) / (exp(lambda) - 1)
    with lambda such that mu = (lambda exp(lambda) - exp(lambda) + 1) / (lambda * (exp(lambda) - 1))
    
  - in practice
    preconstruct array M such that M[mu] = lambda and use it to look up value of lambda for given mu
    
  - compute mu, look-up for lambda, compute f(s) for all s
  - compute CDF F of f
  - compute histogram of input image h(s)
  - compute CDF H of h
  - for each s, do mapping F^-1 (H(s))
*/


void histogram_matching( int nbin, float* h, float* f, int* newval );

void build_target_distrib_bpheme( int m, int nbin, float* f );

void BPHEME( cv::Mat& img, cv::Mat& out );


#endif // _BPHEME_H_

