/*
 *  getsysinfo_win_kernel.cpp
 *
 
 Stub for getting system hardware information from MATLAB on Windows. The information
 returned is the number of cores (logical processors), the memory size in bytes and the 
 maximum size in byte of caches found.
 
 *
 *  Created by Antonio Martinez on 02/04/14.
 *  Copyright 2014 Universidad Almeria / Max Planck Insititute of Biochemistry. All rights reserved.
 *
 */

#include "tsmtv_helpers.h"

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{ 
	int *out1;
	long long int *out2;
	long long int *out3;
	
	/* Create data for holding the output result */
	plhs[0] = mxCreateNumericMatrix( 1, 1, mxINT32_CLASS, mxREAL );
	out1 = (int*)mxGetData( plhs[0] );
	plhs[1] = mxCreateNumericMatrix( 1, 1, mxINT64_CLASS, mxREAL );
	out2 = (long long int*)mxGetData( plhs[1] );
	plhs[2] = mxCreateNumericMatrix( 1, 1, mxINT64_CLASS, mxREAL );
	out3 = (long long int*)mxGetData( plhs[2] );	
	
	/* Ask to computer about itself */
	out1[0] = get_num_processors();
	out2[0] = get_mem_size();
	out3[0] = get_cache_size();
	
	return;
}
