

#ifndef _FHSABP_H_
#define _FHSABP_H_


/*
  Flattest histogram specification with accurate brightness preservation (FHSABP)
  C. Wang J. Peng Z. Ye
  IET Image Process., 2008, Vol. 2, No. 5, pp. 249– 262

  Idea:
  - Compute target distribution and use histogram matchine (aka. histogram specification)
    to compute mapping function from input grayscale to new grayscale value

  - target distribution minimizes squared difference to uniform distribution, subject to
    mean intensity constraint

  Algorithm:
  1) if 0 <= mu < 84.67
     compute x0 = floor(3*mu) +1
     a = (-6 x0 + 12 mu)  / (x0 (x0+1) (x0+2))
     b = (4 x0 - 6 mu +2) / ((x0+1) (x0+2))
     pi = max{ ai + b, 0 (?)  }, i=0,...,255 

  2) if 84.67 <= mu < 170.33
     a = (mu - 127.5) / 1398080
     b = (511 - 3 mu) / 32896
     pi = ai + b, i = 0,...,255

  3) 170.33 < mu
     mu' = 255 - mu
     apply case 1) with mu' 
     invert pi = p(255-i)

 */


#include <opencv2/opencv.hpp>

void build_target_distrib_fhsabp( float mu, float* f );

void FHSABP( cv::Mat& img, cv::Mat& out );


#endif // _FHSABP_H_

