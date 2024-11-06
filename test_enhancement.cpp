

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
using namespace std;

#include "image_enhancement.h"

#include <opencv2/opencv.hpp>
using namespace cv;


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "ahe.h"
#include "lide_simple.h"
#include "lide_mixture.h"

int main( int argc, char* argv[] )
{
	if( argc!=3 )
	{
		cerr<< "Error"<< endl
		    << "Usage : "<< argv[0]<< " img_filename output_prefix"<< endl;
		return 0;
	}
	char* img_filename  = argv[1];
	char* output_prefix = argv[2];
	

	Mat org = imread( img_filename, CV_LOAD_IMAGE_COLOR );
	cerr<< org.cols<< " "<< org.rows<< endl;
	
	Mat out( org.rows, org.cols, CV_8UC3 );
		
	char filename[1000];
	EnhanceParam param;

	// for local enhancement method
	param.d = 200; 
	param.lide_k = 10; // 

	for( int m = 0; m < 12; m++ )
	{
		param.method = m;

		cerr<< "[";
		double t0 = clock();
		
		switch(m)
		{
		case ENHANCE_HE     : color_enhance( org, out, param ); sprintf( filename, "%s_%d_HE.png" , output_prefix, m ); break;
		case ENHANCE_DHE    : color_enhance( org, out, param ); sprintf( filename, "%s_%d_DHE.png", output_prefix, m ); break;
		case ENHANCE_MHE    : color_enhance( org, out, param ); sprintf( filename, "%s_%d_MHE.png", output_prefix, m ); break;
		case ENHANCE_ESIHE  : color_enhance( org, out, param ); sprintf( filename, "%s_%d_ESIHE.png", output_prefix, m ); break;
		case ENHANCE_BPHEME : color_enhance( org, out, param ); sprintf( filename, "%s_%d_BPHEME.png", output_prefix, m ); break;
		case ENHANCE_FHSABP : color_enhance( org, out, param ); sprintf( filename, "%s_%d_FHSABP.png", output_prefix, m ); break;
		case ENHANCE_HEGMM  : color_enhance( org, out, param ); sprintf( filename, "%s_%d_HEGMM.png", output_prefix, m ); break;
		case ENHANCE_AHE    : color_enhance( org, out, param ); sprintf( filename, "%s_%d_AHE_d%d.png"  , output_prefix, m, param.d ); break;
		case ENHANCE_LIDEG  : color_enhance( org, out, param ); sprintf( filename, "%s_%d_LIDEG_d%d.png", output_prefix, m, param.d ); break;
		case ENHANCE_LIDEL  : color_enhance( org, out, param ); sprintf( filename, "%s_%d_LIDEL_d%d.png", output_prefix, m, param.d ); break;
		case ENHANCE_LIDEGMM: color_enhance( org, out, param ); sprintf( filename, "%s_%d_LIDEGMM_d%d_k%d.png", output_prefix, m, param.d, param.lide_k ); break;
		case ENHANCE_LIDELMM: color_enhance( org, out, param ); sprintf( filename, "%s_%d_LIDELMM_d%d_k%d.png", output_prefix, m, param.d, param.lide_k ); break;
		}
		
		double t1 = clock();
		cerr<< "]"<< endl;
		
		imwrite( filename, out );
		
		cout<< "Time "<< filename<< " "<< (double)(t1-t0)/CLOCKS_PER_SEC<< endl;
		
		cerr<< ".";
	}
	cerr<< endl;
        



	// local and brightness-preserving enhancement
	param.target = 1; // BPHEME





	param.d = 200; 

	// 
	param.target = 2; // FHSABP

	// local+mixture methods -> depend on param.d and param.lide_k
	int tabk[] = { 3, 5, 10, 5, 10, 20 };
	// for( int k = 0; k < 3; k++ )
	{
		param.lide_k = 10; // tabk[k];

		for( int m = 10; m < 12; m++ )
		{
			param.method = m;
			
			cerr<< "[";
			double t0 = clock();
			
			switch(m)
			{
			case ENHANCE_LIDEGMM: color_enhance( org, out, param ); sprintf( filename, "%s_%d_LIDEGMM_d%d_k%d_FSHABP.png", output_prefix, m, param.d, param.lide_k ); break;
			case ENHANCE_LIDELMM: color_enhance( org, out, param ); sprintf( filename, "%s_%d_LIDELMM_d%d_k%d_FSHABP.png", output_prefix, m, param.d, param.lide_k ); break;
			}
		
			double t1 = clock();
			cerr<< "]"<< endl;

			imwrite( filename, out );

			cout<< "Time "<< filename<< " "<< (double)(t1-t0)/CLOCKS_PER_SEC<< endl;

			cerr<< ".";

		}
	}
	cerr<< endl;

	return 0;
}

