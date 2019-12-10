/*
 *  getsysinfo_mac_kernel.h
 *
 
 Kernel for getting system hardware information from MATLAB in a Mac OS. The information
 returned is the number of cores, the memory size and the L2 cache size (in this order).
 
 *
 *  Created by Antonio Martinez on 1/10/13.
 *  Copyright 2013 Universidad Almeria. All rights reserved.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mex.h>

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{ 
	int mib[2];
	int datum;
	size_t len, len64;
	long long int dat64_1, dat64_2;
	int* out1 = &datum;
	long long int* out2 = &dat64_1;
	long long int* out3 = &dat64_2;
	
	// Create data for holding the output result
	plhs[0] = mxCreateNumericMatrix( 1, 1, mxINT32_CLASS, mxREAL );
	out1 = mxGetPr( plhs[0] );
	plhs[1] = mxCreateNumericMatrix( 1, 1, mxINT64_CLASS, mxREAL );
	out2 = mxGetPr( plhs[1] );
	plhs[2] = mxCreateNumericMatrix( 1, 1, mxINT64_CLASS, mxREAL );
	out3 = mxGetPr( plhs[2] );	
	
	// Ask to computer about itself
	datum = -1;
	len = sizeof( datum );
	sysctlbyname("hw.activecpu", out1, &len, NULL, 0);
	dat64_1 = -1;
	len64 = sizeof( dat64_1 );
	sysctlbyname("hw.memsize", out2, &len64, NULL, 0);		
	dat64_2 = -1;
	sysctlbyname("hw.l2cachesize", out3, &len64, NULL, 0);
	
	return;
}
