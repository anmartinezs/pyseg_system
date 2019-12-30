/*
 *  nonmaxsup_stub.c
 
 Stub for manifold detection by non-maximum suppresion criteria, input data must be 
 one-dimensional arrays
 
 *  
 *
 *  Created by Antonio Martinez on 1/14/13.
 *  Copyright 2013 Universidad Almeria. All rights reserved.
 *
 */

#include <mex.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>

// Constants
#define INTER_FACTOR .71
#define CACHE_USED .33
#define BUFFER_SIZE 1024
#define BYTE_PER_KBYTE 1024
#define ERROR_FILE_READ -1
#define ERROR_CACHE_NOT_FOUND -2
#define ERROR_MEM_NOT_FOUND -3

// Global Variables
int sq, lq, ld, task_size;
long long int *M;
double *I, *Vx, *Vy, *Vz;
unsigned char *F;
pthread_mutex_t mutex;

// Global functions
void* look_neigbourhood( void* ptr );
long long int get_cache_size();

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
	unsigned int* dim;
	int i, nta, nth, type, num_threads;
	mwSize m, mh;
	long long int dat64;
	pthread_t* threads;
	
	// Check for proper number of arguments
    if (nrhs != 6) { 
		mexErrMsgTxt("nonmaxsup_stub.c: Six input arguments required."); 
		return;
    } else if (nlhs > 1) {
		mexErrMsgTxt("nonmaxsup_stub.c: Too many output arguments."); 
		return;
    } 
	
	// Get computer information to set the number of thread and the size of buffers
	num_threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
	if (num_threads<1) {
		mexErrMsgTxt("nonmaxsup_stub.cpp: No active cpu detected.");
		return;
	}
	dat64 = get_cache_size();
	if (dat64<1) {
		mexErrMsgTxt("nonmaxsup_stub.cpp: No Cache detected.");
		return;
	}
	task_size = ceil( float(CACHE_USED*dat64) / 18 );
	
	// Check the format of input data are OK
	for (i=0; i<4; i++) {
		if (!mxIsComplex(prhs[i])) {
			if (i==0) {
				m = mxGetM( prhs[i] );
			}else {
				mh = mxGetM( prhs[i] );
				if (m!=mh) {
					mexErrMsgTxt("nonmaxsup_stub.c: Dimensions mismatch."); 
					return;
				}else if (mxGetClassID(prhs[i])!=mxDOUBLE_CLASS) {
					mexErrMsgTxt("nonmaxsup_stub.c: Input data must be double."); 
					return;
				}
			}
		}else {
			mexErrMsgTxt("nonmaxsup_stub.c: Input data must be real."); 
			return;
		}
	}
	if (mxGetClassID(prhs[4])!=mxUINT64_CLASS) {
		mexErrMsgTxt("nonmaxsup_stub.c: Mask data must be 64 bits integer."); 
		return;
	}
	mh = mxGetN(prhs[4]); 
	if (mh>m) {
		mexErrMsgTxt("nonmaxsup_stub.c: Mask dimensions mismatch."); 
		return;
	}
	if (mxGetClassID(prhs[5])!=mxUINT32_CLASS) {
		mexErrMsgTxt("nonmaxsup_stub.c: Tomogram dimension must be 32 bits integer."); 
		return;
	}
	ld = mh;
	
	// Create the array for holding the output result
	plhs[0] = mxCreateLogicalMatrix( m, 1 );					
	
	// Assign pointers to data
	I = mxGetPr( prhs[0] );
	Vx = mxGetPr( prhs[1] );
	Vy = mxGetPr( prhs[2] );
	Vz = mxGetPr( prhs[3] );
	M = (long long int*)mxGetData( prhs[4] );
	dim = (unsigned int*)mxGetData(prhs[5]);
	F = (unsigned char*)mxGetData( plhs[0] );
	
	// Set pointer for initial splitting
	nta = (float)m / task_size;
	nta = ceil( nta );
	nth = num_threads;
	if (nta<nth) {
		nth = nta;
	}	
	
	// Throw the workers
	lq = m;
	sq = 0; // Task queue initialization
	if (pthread_mutex_init( &mutex, NULL ) ){
		mexErrMsgTxt("nonmaxsup_stub.c: Error creating the mutex.");
		return;
	}
	threads = (pthread_t*)malloc( nth*sizeof(pthread_t) );
	for (i=0; i<nth; i++) {
		// Update process queue pointers
		if (pthread_create(&threads[i],NULL,&look_neigbourhood,(void*)dim)) {
			mexErrMsgTxt("nonmaxsup_stub.c: Error creating a thread.");
			return;
		}
	}
	
	// Wait for all workers
	for (i=0; i<nth; i++) {
		if (pthread_join(threads[i],NULL)) {
			mexErrMsgTxt("nonmaxsup_stub.c: Error waiting the thread termination.");
			return;
		}
	}
	
	return;
}

// Thread for measuring the intermediate neighbour value
void* look_neigbourhood( void* ptr ){

	unsigned long long int i, j, k;
	unsigned int mx, my;
	unsigned int *dim;
	int sz, start, end;
	double lv, hold, kx, ky, kz;
	double* A[8];
	double* B[8];
	double* Va;
	double* Vb;
	double** K;
	unsigned char lock = 0x01;
	dim = (unsigned int*)ptr;
	mx = dim[0];
	my = dim[1];
	
	// Buffers initialization
	sz = task_size * sizeof(double);
	for (i=0; i<8; i++) {
		A[i] = (double*)malloc( sz );
		B[i] = (double*)malloc( sz );
	}	
	Va = (double*)malloc( sz );
	Vb = (double*)malloc( sz );	
	K = (double**)malloc( 3*sizeof(double*) );
	for (i=0; i<3; i++) {
		K[i] = (double*)malloc( sz );
	}
	
	// Task loop
	do{
		// Update pointers
		pthread_mutex_lock( &mutex );
		start = sq;
		sq = start + task_size;	
		if (sq>=ld) {
			sq = ld;
			lock = 0x00;
		}
		end = sq;
		pthread_mutex_unlock( &mutex );		
		
		// Prepare data for every coordinate
		j = 0;
		for (k=start; k<end; k++) {
			i = M[k];			
			lv = I[i];
			K[0][j] = fabs( Vx[i] * INTER_FACTOR );
			K[1][j] = fabs( Vy[i] * INTER_FACTOR );
			K[2][j] = fabs( Vz[i] * INTER_FACTOR );
			A[0][j] = lv;
			B[0][j] = lv;
			if (Vx[i]>=0) {
				A[1][j] = I[i+mx*my];
				B[1][j] = I[i-mx*my];
				if ( (Vy[i]>=0) && (Vz[i]>=0) ) {
					A[2][j] = I[i+mx];
					A[3][j] = I[i+mx*(my+1)];
					A[4][j] = I[i+1];
					A[5][j] = I[i+mx*my+1];
					A[6][j] = I[i+mx+1];
					A[7][j] = I[i+mx*(my+1)+1];
					B[2][j] = I[i-mx];
					B[3][j] = I[i-mx*(my+1)];
					B[4][j] = I[i-1];
					B[5][j] = I[i-mx*my-1];
					B[6][j] = I[i-mx-1];					
					B[7][j] = I[i-mx*(my+1)-1];
				}else if ( (Vy[i]<0) && (Vz[i]>=0) ) {
					A[2][j] = I[i-mx];
					A[3][j] = I[i+mx*(my-1)];
					A[4][j] = I[i+1];
					A[5][j] = I[i+mx*my+1];
					A[6][j] = I[i-mx+1];
					A[7][j] = I[i+mx*(my-1)+1];
					B[2][j] = I[i+mx];
					B[3][j] = I[i-mx*(my-1)];
					B[4][j] = I[i-1];
					B[5][j] = I[i-mx*my-1];
					B[6][j] = I[i+mx-1];
					B[7][j] = I[i-mx*(my-1)-1];
				}else if ( (Vy[i]>=0) && (Vz[i]<0) ) {
					A[2][j] = I[i+mx];
					A[3][j] = I[i+mx*(my+1)];
					A[4][j] = I[i+1];
					A[5][j] = I[i+mx*my+1];
					A[6][j] = I[i+mx+1];
					A[7][j] = I[i+mx*(my+1)+1];
					B[2][j] = I[i-mx];
					B[3][j] = I[i-mx*(my+1)];
					B[4][j] = I[i-1];
					B[5][j] = I[i-mx*my-1];
					B[6][j] = I[i-mx-1];
					B[7][j] = I[i-mx*(my+1)-1];
				}else {
					A[2][j] = I[i-mx];
					A[3][j] = I[i+mx*(my-1)];
					A[4][j] = I[i+1];
					A[5][j] = I[i+mx*my+1];
					A[6][j] = I[i-mx+1];
					A[7][j] = I[i+mx*(my-1)+1];
					B[2][j] = I[i+mx];
					B[3][j] = I[i-mx*(my-1)];
					B[4][j] = I[i-1];
					B[5][j] = I[i-mx*my-1];
					B[6][j] = I[i+mx-1];
					B[7][j] = I[i-mx*(my-1)-1];
				}
			}else {
				A[1][j] = I[i-mx*my];
				B[1][j] = I[i+mx*my];
				if ( (Vy[i]>=0) && (Vz[i]>=0) ) {
					A[2][j] = I[i+mx];
					A[3][j] = I[i-mx*(my-1)];
					A[4][j] = I[i-1];
					A[5][j] = I[i-mx*my-1];
					A[6][j] = I[i+mx-1];
					A[7][j] = I[i-mx*(my-1)-1];
					B[2][j] = I[i-mx];
					B[3][j] = I[i+mx*(my-1)];
					B[4][j] = I[i+1];
					B[5][j] = I[i+mx*my+1];
					B[6][j] = I[i-mx+1];
					B[7][j] = I[i+mx*(my-1)+1];
				}else if ( (Vy[i]<0) && (Vz[i]>=0) ) {
					A[2][j] = I[i-mx];
					A[3][j] = I[i-mx*(my+1)];
					A[4][j] = I[i-1];
					A[5][j] = I[i-mx*my-1];
					A[6][j] = I[i-mx-1];
					A[7][j] = I[i-mx*(my+1)-1];
					B[2][j] = I[i+mx];
					B[3][j] = I[i+mx*(my+1)];
					B[4][j] = I[i+1];
					B[5][j] = I[i+mx*my+1];
					B[6][j] = I[i+mx+1];
					B[7][j] = I[i+mx*(my+1)+1];
				}else if ( (Vy[i]>=0) && (Vz[i]<0) ) {
					A[2][j] = I[i+mx];
					A[3][j] = I[i-mx*(my-1)];
					A[4][j] = I[i-1];
					A[5][j] = I[i-mx*my-1];
					A[6][j] = I[i+mx-1];
					A[7][j] = I[i-mx*(my-1)-1];
					B[2][j] = I[i-mx];
					B[3][j] = I[i+mx*(my-1)];
					B[4][j] = I[i+1];
					B[5][j] = I[i+mx*my+1];
					B[6][j] = I[i-mx+1];
					B[7][j] = I[i+mx*(my-1)+1];
				}else {
					A[2][j] = I[i-mx];
					A[3][j] = I[i-mx*(my+1)];
					A[4][j] = I[i-1];
					A[5][j] = I[i-mx*my-1];
					A[6][j] = I[i-mx-1];					
					A[7][j] = I[i-mx*(my+1)-1];
					B[2][j] = I[i+mx];
					B[3][j] = I[i+mx*(my+1)];
					B[4][j] = I[i+1];
					B[5][j] = I[i+mx*my+1];
					B[6][j] = I[i+mx+1];
					B[7][j] = I[i+mx*(my+1)+1];
				}
			}				
			j++;
		}
		
		// Trilinear interpolation
		for (j=0; j<(end-start); j++) {
			kx = K[0][j];
			ky = K[1][j];
			kz = K[2][j];
			hold = A[0][j]*(1-kx)*(1-ky)*(1-kz);
			hold = hold + A[4][j]*kx*(1-ky)*(1-kz);
			hold = hold + A[2][j]*(1-kx)*ky*(1-kz);
			hold = hold + A[1][j]*(1-kx)*(1-ky)*kz;
			hold = hold + A[5][j]*kx*(1-ky)*kz;
			hold = hold + A[3][j]*(1-kx)*ky*kz;
			hold = hold + A[6][j]*kx*ky*(1-kz);
			Va[j] = hold + A[7][j]*kx*ky*kz;			
		}
		for (j=0; j<(end-start); j++) {
			kx = K[0][j];
			ky = K[1][j];
			kz = K[2][j];
			hold = B[0][j]*(1-kx)*(1-ky)*(1-kz);
			hold = hold + B[4][j]*kx*(1-ky)*(1-kz);		
			hold = hold + B[2][j]*(1-kx)*ky*(1-kz);		
			hold = hold + B[1][j]*(1-kx)*(1-ky)*kz;			
			hold = hold + B[5][j]*kx*(1-ky)*kz;		
			hold = hold + B[3][j]*(1-kx)*ky*kz;		
			hold = hold + B[6][j]*kx*ky*(1-kz);		
			Vb[j] = hold + B[7][j]*kx*ky*kz;	
		}
		
		// Mark local maxima
		j = 0;
		for (k=start; k<end; k++) {
			i = M[k];
			lv = I[i];			
			if ( (lv>Va[j]) && (lv>Vb[j]) ) {
				F[i] = 0x01;			
			}
			j++;
		}
		
	}while(lock);
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
