

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
using namespace std;

#include "image_enhancement.h"

#include <opencv2/opencv.hpp>
using namespace cv;


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
string method_name( int m )
{
    switch(m)
    {
    case ENHANCE_HE: return "HE";
    case ENHANCE_AHE: return "AHE";
    case ENHANCE_DHE: return "DHE";
    case ENHANCE_MHE: return "MHE";
    case ENHANCE_ESIHE: return "ESIHE";
    case ENHANCE_BPHEME: return "BPHEME";
    case ENHANCE_FHSABP: return "FHSABP";
    case ENHANCE_HEGMM: return "HEGMM";
    case ENHANCE_LIDES:  return "LIDE (simple)";
    case ENHANCE_LIDEMM: return "LIDE (mixture)";
    case ENHANCE_PS: return "PS";
    }
    return "Unknown!!";
}



// We assume that the caller call correctly!! 
int main( int argc, char* argv[] )
{
    char *img_filename = NULL;
    char *out_filename = NULL;

    EnhanceParam param;
    for( int i = 1; i < argc; i++ )
    {
	if (argv[i][0]!='-')
	{
	    cerr<< "Error : invalid argument"<< endl;
	    return 0;
	}
	if (strcmp(argv[i], "-in") == 0)
	{	 
	    img_filename = argv[i+1];
	    i++;
	}
	else if (strcmp(argv[i], "-out") == 0)
	{
	    out_filename = argv[i+1];
	    i++;
	}
	else if (strcmp(argv[i], "-method") == 0)
	{
	    param.method = atoi(argv[i+1]);
	    i++;
	}
	else if (strcmp(argv[i], "-AHE:clipping")==0)
	{
	    param.ahe_clipping = atof(argv[i+1]);
	    i++;
	}
	else if (strcmp(argv[i], "-AHE:rescale_intensity")==0)
	{
	    param.ahe_rescale_intensity = (atoi(argv[i+1])==1);
	    i++;
	}
	else if (strcmp(argv[i], "-DHE:min_length")==0)
	{
	    param.dhe_min_length = atoi(argv[i+1]);
	    i++;
	}
	else if (strcmp(argv[i], "-MHE:mmin")==0)
	{
	    param.mhe_mmin = atoi(argv[i+1]);
	    i++;
	}
	else if (strcmp(argv[i], "-MHE:mmax")==0)
	{
	    param.mhe_mmax = atoi(argv[i+1]);
	    i++;
	}
	else if (strcmp(argv[i], "-MHE:rho")==0)
	{
	    param.mhe_rho = atof(argv[i+1]);
	    i++;
	}
	else if (strcmp(argv[i], "-HEGMM:k")==0)
	{
	    param.hegmm_k = atoi(argv[i+1]);
	    i++;
	}
	else if (strcmp(argv[i], "-LIDE:mod")==0)
	{
	    param.lide_mod = atoi(argv[i+1]);
	    i++;
	}
	else if (strcmp(argv[i], "-LIDE:k")==0)
	{
	    param.lide_k = atoi(argv[i+1]);
	    i++;
	}
	else if (strcmp(argv[i], "-LIDE:sigma_min")==0)
	{
	    param.lide_sigma_min = atoi(argv[i+1]);
	    i++;
	}
	else if (strcmp(argv[i], "-PS:ps_phi")==0)
	{
	    param.ps_phi = atoi(argv[i+1]);
	    i++;
	}
	else if (strcmp(argv[i], "-ALL:target")==0)
	{
	    param.target = atoi(argv[i+1]);
	    i++;
	}
	else if (strcmp(argv[i], "-ALL:d")==0)
	{
	    param.d = atoi(argv[i+1]);
	    i++;
	}
    }

    if (img_filename==NULL || out_filename==NULL)
    {
	cerr<< "Error: empty files"<< endl;
	return 0;
    }
    cerr<< "Input filename: "<< img_filename<< endl
	<< "Output filename: "<< out_filename<< endl
	<< "Method "<< param.method<< " ("<< method_name(param.method)<< ")"<< endl
	<< "d "<< param.d<< endl
	<< "target "<< param.target<< endl
	<< "AHE:clipping "<< param.ahe_clipping<< endl
	<< "AHE:rescale_intensity "<< param.ahe_rescale_intensity<< endl
	<< "DHE:min_length "<< param.dhe_min_length<< endl
	<< "MHE:mmin "<< param.mhe_mmin<< endl
	<< "MHE:mmax "<< param.mhe_mmax<< endl
	<< "MHE:rho "<< param.mhe_rho<< endl
	<< "HEGMM:k "<< param.hegmm_k<< endl
	<< "LIDE:mod "<< param.lide_mod<< endl
	<< "LIDE:k "<< param.lide_k<< endl
	<< "LIDE:sigma_min "<< param.lide_sigma_min<< endl
	<< "PS:phi "<< param.ps_phi<< endl
	<< "=============================================="<< endl<< endl;
    
    Mat org = imread( img_filename, CV_LOAD_IMAGE_COLOR );
    Mat out( org.rows, org.cols, CV_8UC3 );

    color_enhance( org, out, param );

    imwrite( out_filename, out );

    return 0;
    
}

