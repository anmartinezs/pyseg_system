/*
 *  nonmaxsup_stub_single.c
 
	Stub for manifold detection by non-maximum suppresion criteria, input data must be 
	one-dimensional arrays, input/output data are single precision float.
 
 
 *  
 *
 *  Created by Antonio Martínez on 1/22/13.
 *  Copyright 2013 Universidad Almeria. All rights reserved.
 *
 */

#include <mex.h>
#include <math.h>
#include <pthread.h>
#include <sys/sysctl.h>

// Constants
#define CACHE_L2_USED .33
#define INTER_FACTOR .71

// Global Variables
int sq, lq, ld, num_threads, task_size;
long long int *M;
float *I, *Vx, *Vy, *Vz;
unsigned char *F;
pthread_mutex_t mutex;

// Global functions
void* look_neigbourhood( void* ptr );

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
	unsigned int* dim;
	int i, nta, nth, type;
	mwSize m, mh;
	size_t len, len64;
	long long int dat64;
	pthread_t* threads;
	
	// Check for proper number of arguments
    if (nrhs != 6) { 
		mexErrMsgTxt("nonmaxsup_stub_single.c: Six input arguments required."); 
		return;
    } else if (nlhs > 1) {
		mexErrMsgTxt("nonmaxsup_stub_single.c: Too many output arguments."); 
		return;
    } 
	
	// Get computer information to set the number of thread and the size of buffers
	num_threads = -1;
	len = sizeof( num_threads );
	sysctlbyname("hw.activecpu", &num_threads, &len, NULL, 0);
	if (num_threads<1) {
		mexErrMsgTxt("nonmaxsup_stub_single.c: No active cpu detected.");
		return;
	}
	dat64 = -1;
	len64 = sizeof( dat64 );
	sysctlbyname("hw.l2cachesize", &dat64, &len64, NULL, 0);
	if (dat64<1) {
		mexErrMsgTxt("nonmaxsup_stub_single.c: No Cache L2 detected.");
		return;
	}
	task_size = (int)ceil( ((double)(CACHE_L2_USED*dat64)) / 6. );
	
	// Check the format of input data are OK
	for (i=0; i<4; i++) {
		if (!mxIsComplex(prhs[i])) {
			if (i==0) {
				m = mxGetM( prhs[i] );
			}else {
				mh = mxGetM( prhs[i] );
				if (m!=mh) {
					mexErrMsgTxt("nonmaxsup_stub_single.c: Dimensions mismatch."); 
					return;
				}else if (mxGetClassID(prhs[i])!=mxSINGLE_CLASS) {
					mexErrMsgTxt("nonmaxsup_stub_single.c: Input data must be double."); 
					return;
				}
			}
		}else {
			mexErrMsgTxt("nonmaxsup_stub_single.c: Input data must be real."); 
			return;
		}
	}
	if (mxGetClassID(prhs[4])!=mxUINT64_CLASS) {
		mexErrMsgTxt("nonmaxsup_stub_single.c: Mask data must be 64 bits integer."); 
		return;
	}
	mh = mxGetN(prhs[4]); 
	if (mh>m) {
		mexErrMsgTxt("nonmaxsup_stub_single.c: Mask dimensions mismatch."); 
		return;
	}
	if (mxGetClassID(prhs[5])!=mxUINT32_CLASS) {
		mexErrMsgTxt("nonmaxsup_stub_single.c: Tomogram dimension must be 32 bits integer."); 
		return;
	}
	ld = mh;
	
	// Create the array for holding the output result
	plhs[0] = mxCreateLogicalMatrix( m, 1 );					
	
	// Assign pointers to data
	I = (float*)mxGetData( prhs[0] );
	Vx = (float*)mxGetData( prhs[1] );
	Vy = (float*)mxGetData( prhs[2] );
	Vz = (float*)mxGetData( prhs[3] );
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
	float lv, hold, kx, ky, kz;
	float* A[8];
	float* B[8];
	float* Va;
	float* Vb;
	float** K;
	unsigned char lock = 0x01;
	dim = (unsigned int*)ptr;
	mx = dim[0];
	my = dim[1];
	
	// Buffers initialization
	sz = task_size * sizeof(float);
	for (i=0; i<8; i++) {
		A[i] = (float*)malloc( sz );
		B[i] = (float*)malloc( sz );
	}	
	Va = (float*)malloc( sz );
	Vb = (float*)malloc( sz );	
	K = (float**)malloc( 3*sizeof(float*) );
	for (i=0; i<3; i++) {
		K[i] = (float*)malloc( sz );
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


