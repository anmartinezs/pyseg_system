/*
 *  getsysinfo_linux.h
 *
 
 Stub for getting system hardware information from MATLAB in a Linux. The information
 returned is the number of cores, the memory size and the L2 cache size (in this order).
 
 *
 *  Created by Antonio Martinez on 1/10/13.
 *  Copyright 2013 Universidad Almeria. All rights reserved.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <mex.h>

#define BUFFER_SIZE 1024
#define BYTE_PER_KBYTE 1024

#define ERROR_FILE_READ -1
#define ERROR_CACHE_NOT_FOUND -2
#define ERROR_MEM_NOT_FOUND -3

long long int get_cache_size(); 
long long int get_mem_size(); 

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
	out1[0] = (int)sysconf(_SC_NPROCESSORS_ONLN);
	out2[0] = get_mem_size();
	out3[0] = get_cache_size();
	
	return;
}

/* Get the cache size found in the first processor listed in /proc/cpuinfo in bytes */
long long int get_cache_size() 
{
	FILE* fp; 
	char *match; 
	char buffer[BUFFER_SIZE];
	size_t bytes_read; 
	long int cache_size; 
	
	fp = fopen( "/proc/cpuinfo", "r" );
	bytes_read = fread(buffer, 1, BUFFER_SIZE, fp); 
	fclose (fp); 
	if( bytes_read <= 0 ) 
		return ERROR_FILE_READ;
	buffer[bytes_read] == '\0';
	match = strstr( buffer, "cache size" ); 
	if (match == NULL) 
		return ERROR_CACHE_NOT_FOUND;  
	sscanf( match, "cache size : %ld", &cache_size ); 
	
	return ((long long int)cache_size) * BYTE_PER_KBYTE; 
}

/* Get the memory size found in /proc/meminfo in bytes */
long long int get_mem_size() 
{
	FILE* fp; 
	char *match; 
	char buffer[BUFFER_SIZE];
	size_t bytes_read; 
	long int mem_size; 
	
	fp = fopen( "/proc/meminfo", "r" );
	bytes_read = fread(buffer, 1, BUFFER_SIZE, fp); 
	fclose (fp); 
	if( bytes_read <= 0 ) 
		return ERROR_FILE_READ;
	buffer[bytes_read] == '\0';
	match = strstr( buffer, "MemTotal" ); 
	if (match == NULL) 
		return ERROR_MEM_NOT_FOUND;  
	sscanf( match, "MemTotal : %ld", &mem_size ); 
	
	return ((long long int)mem_size) * BYTE_PER_KBYTE; 
}