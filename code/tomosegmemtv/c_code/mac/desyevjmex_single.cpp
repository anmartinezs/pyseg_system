/*
 *  desyevjmex_single.cpp
 
 Stub for calculating the eigenvectors and eigenvalues from directional derivatives (reshaped
 into arrays), input/output data are single precision float. Jacobi Kopp2008.
 
 *  
 *
 *  Created by Antonio Martínez on 1/4/13.
 *  Copyright 2013 Universidad Almeria. All rights reserved.
 *
 */

#include <mex.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <sys/sysctl.h>

// Constants
#define M_SQRT3    1.73205080756887729352744634151   // sqrt(3)
#define CACHE_L2_USED .33

// Macros
#define SQR(x)      ((x)*(x)) 

int sq, lq, ld, num_threads, task_size;
float *Ixx, *Iyy, *Izz, *Ixy, *Ixz, *Iyz;
float *L1, *L2, *L3, *V1x, *V1y, *V1z, *V2x, *V2y, *V2z, *V3x, *V3y, *V3z;
pthread_mutex_t mutex;

int dsyevj3(float A[3][3], float Q[3][3], float w[3]);
void* desyevj3stub( void* ptr );

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{ 
	mwSize m;
	int nta, nth;
	pthread_t* threads;
	size_t len, len64;
	long long int dat64;	
	
	// Check for proper number of arguments
    if (nrhs != 6) { 
		mexErrMsgTxt("desyevvmex_single.cpp: Six input arguments required."); 
		return;
    } else if (nlhs > 12) {
		mexErrMsgTxt("desyevvmex_single.cpp: Too many output arguments."); 
		return;
    }
	
	// Get computer information to set the number of thread and the size of buffers
	num_threads = -1;
	len = sizeof( num_threads );
	sysctlbyname("hw.activecpu", &num_threads, &len, NULL, 0);
	if (num_threads<1) {
		mexErrMsgTxt("desyevvmex.cpp: No active cpu detected.");
		return;
	}
	dat64 = -1;
	len64 = sizeof( dat64 );
	sysctlbyname("hw.l2cachesize", &dat64, &len64, NULL, 0);
	if (dat64<1) {
		mexErrMsgTxt("desyevvmex_single.cpp: No Cache L2 detected.");
		return;
	}
	task_size = ceil( float(CACHE_L2_USED*dat64) / 18 );	
	
	// Check the dimensions of the input arrays
    m = mxGetM(prhs[0]);
	ld = m;
	if ( (m!=mxGetM(prhs[1])) || (m!=mxGetM(prhs[2])) || (m!=mxGetM(prhs[3])) || 
		(m!=mxGetM(prhs[4])) || (m!=mxGetM(prhs[5])) ) {
		mexErrMsgTxt("desyevvmex_single.cpp: Dimensions mismatch."); 
		return;
	}
    if ( (mxGetClassID(prhs[0])!=mxSINGLE_CLASS) || mxIsComplex(prhs[0]) || 
		(mxIsDouble(prhs[1])!=mxSINGLE_CLASS) || mxIsComplex(prhs[1]) ||
		(mxIsDouble(prhs[2])!=mxSINGLE_CLASS) || mxIsComplex(prhs[2]) ||
		(mxIsDouble(prhs[3])!=mxSINGLE_CLASS) || mxIsComplex(prhs[3]) ||
		(mxIsDouble(prhs[4])!=mxSINGLE_CLASS) || mxIsComplex(prhs[4]) ||
		(mxIsDouble(prhs[5])!=mxSINGLE_CLASS) || mxIsComplex(prhs[5]) ){ 
		mexErrMsgTxt("desyevvmex_single.cpp: Requires single precision real data."); 
		return;
	} 
	
	// Create the arrays for holding the output results
    plhs[0] = mxCreateNumericMatrix( m, 1, mxSINGLE_CLASS, mxREAL );
	plhs[1] = mxCreateNumericMatrix( m, 1, mxSINGLE_CLASS, mxREAL );
	plhs[2] = mxCreateNumericMatrix( m, 1, mxSINGLE_CLASS, mxREAL );
	plhs[3] = mxCreateNumericMatrix( m, 1, mxSINGLE_CLASS, mxREAL );
	plhs[4] = mxCreateNumericMatrix( m, 1, mxSINGLE_CLASS, mxREAL );
    plhs[5] = mxCreateNumericMatrix( m, 1, mxSINGLE_CLASS, mxREAL );	
    plhs[6] = mxCreateNumericMatrix( m, 1, mxSINGLE_CLASS, mxREAL );
	plhs[7] = mxCreateNumericMatrix( m, 1, mxSINGLE_CLASS, mxREAL );
	plhs[8] = mxCreateNumericMatrix( m, 1, mxSINGLE_CLASS, mxREAL );
	plhs[9] = mxCreateNumericMatrix( m, 1, mxSINGLE_CLASS, mxREAL );
	plhs[10] = mxCreateNumericMatrix( m, 1, mxSINGLE_CLASS, mxREAL );
    plhs[11] = mxCreateNumericMatrix( m, 1, mxSINGLE_CLASS, mxREAL );
	
	// Assign pointers to data
	Ixx = (float*)mxGetData( prhs[0] );
	Iyy = (float*)mxGetData( prhs[1] );
	Izz = (float*)mxGetData( prhs[2] );	
	Ixy = (float*)mxGetData( prhs[3] );
	Ixz = (float*)mxGetData( prhs[4] );
	Iyz = (float*)mxGetData( prhs[5] );
	L1 = (float*)mxGetData( plhs[0] );
	L2 = (float*)mxGetData( plhs[1] );
	L3 = (float*)mxGetData( plhs[2] );	
	V1x =(float*)mxGetData( plhs[3] );
	V1y = (float*)mxGetData( plhs[4] );
	V1z = (float*)mxGetData( plhs[5] );
	V2x = (float*)mxGetData( plhs[6] );
	V2y = (float*)mxGetData( plhs[7] );
	V2z = (float*)mxGetData( plhs[8] );
	V3x = (float*)mxGetData( plhs[9] );
	V3y = (float*)mxGetData( plhs[10] );
	V3z = (float*)mxGetData( plhs[11] );
	
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
		mexErrMsgTxt("desyevvmex_single.cpp: Error creating the mutex.");
		return;
	}
	threads = (pthread_t*)malloc( nth*sizeof(pthread_t) );
	for (int i=0; i<nth; i++) {
		// Update process queue pointers
		if (pthread_create(&threads[i],NULL,&desyevj3stub,NULL)) {
			mexErrMsgTxt("desyevvmex_single.cpp: Error creating a thread.");
			return;
		}
	}
	
	// Wait for all workers
	for (int i=0; i<nth; i++) {
		if (pthread_join(threads[i],NULL)) {
			mexErrMsgTxt("desyevvmex_single.cpp: Error waiting the thread termination.");
			return;
		}
	} 
	
	return;
}	

// Thread for calling the function for doing the calculations
void* desyevj3stub( void* ptr ){
	
	int start, end;
	float A[3][3];
	float Q[3][3];
	float w[3];
	bool lock = true;
	
	do{
		// Update pointers
		pthread_mutex_lock( &mutex );
		start = sq;
		sq = start + task_size;
		if (sq>=ld) {
			sq = ld;
			lock = false;
		}
		end = sq;
		pthread_mutex_unlock( &mutex );
		
		// Precesing task
		for (int i=start; i<end; i++) {
			// Fill stub data
			A[0][0] = Ixx[i];
			A[0][1] = Ixy[i];
			A[0][2] = Ixz[i];
			A[1][0] = Ixy[i];
			A[1][1] = Iyy[i];
			A[1][2] = Iyz[i];
			A[2][0] = Ixz[i];
			A[2][1] = Iyz[i];
			A[2][2] = Izz[i];
			// Eigemproblem computation
			dsyevj3( A, Q, w );					
			// Fill output arrays with the results
			V1x[i] = Q[0][0];
			V1y[i] = Q[1][0];
			V1z[i] = Q[2][0];
			V2x[i] = Q[0][1];
			V2y[i] = Q[1][1];
			V2z[i] = Q[2][1];
			V3x[i] = Q[0][2];
			V3y[i] = Q[1][2];
			V3z[i] = Q[2][2];
			L1[i] = w[0];
			L2[i] = w[1];
			L3[i] = w[2];			
		}
	}while(lock);
}

// ----------------------------------------------------------------------------
int dsyevj3(float A[3][3], float Qo[3][3], float w[3])
// ----------------------------------------------------------------------------
// Calculates the eigenvalues and normalized eigenvectors of a symmetric 3x3
// matrix A using the Jacobi algorithm.
// The upper triangular part of A is destroyed during the calculation,
// the diagonal elements are read but not destroyed, and the lower
// triangular elements are not referenced at all.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The symmetric input matrix
//   Q: Storage buffer for eigenvectors
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error (no convergence)
// ----------------------------------------------------------------------------
{
	const int n = 3;
	float sd, so;                  // Sums of diagonal resp. off-diagonal elements
	float s, c, t;                 // sin(phi), cos(phi), tan(phi) and temporary storage
	float g, h, z, theta;          // More temporary storage
	float thresh;
	float hold;
	int id0, id1, id2;
	float Q[3][3];
	
	// Initialize Q to the identitity matrix
#ifndef EVALS_ONLY
	for (int i=0; i < n; i++)
	{
		Q[i][i] = 1.0;
		for (int j=0; j < i; j++)
			Q[i][j] = Q[j][i] = 0.0;
	}
#endif
	
	// Initialize w to diag(A)
	for (int i=0; i < n; i++)
		w[i] = A[i][i];
	
	// Calculate SQR(tr(A))  
	sd = 0.0;
	for (int i=0; i < n; i++)
		sd += fabs(w[i]);
	sd = SQR(sd);
	
	// Main iteration loop
	for (int nIter=0; nIter < 50; nIter++)
	{
		// Test for convergence 
		so = 0.0;
		for (int p=0; p < n; p++)
			for (int q=p+1; q < n; q++)
				so += fabs(A[p][q]);
		if (so == 0.0){
			// Ordering eigenvalues (buble sort)
			id0 = 0;
			id1 = 1;
			id2 = 2;
			if ( fabs(w[1]) > fabs(w[0]) ) {
				hold = w[1];
				w[1] = w[0];
				w[0] = hold;
				id0 = 1;
				id1 = 0;
			}
			if ( fabs(w[2]) > fabs(w[1]) ) {
				hold = w[2];
				w[2] = w[1];
				w[1] = hold;
				hold = id1;
				id1 = 2;
				id2 = id1;
			}
			if ( fabs(w[1]) > fabs(w[0]) ) {
				hold = w[1];
				w[1] = w[0];
				w[0] = hold;
				hold = id0;
				id0 = id1;
				id1 = hold;
			}			
			
			// Reorder eigenvectors
			Qo[0][0] = Q[0][id0];	
			Qo[1][0] = Q[1][id0];
			Qo[2][0] = Q[2][id0];	
			Qo[0][1] = Q[0][id1];	
			Qo[1][1] = Q[1][id1];
			Qo[2][1] = Q[2][id1];		
			Qo[0][2] = Q[0][id2];	
			Qo[1][2] = Q[1][id2];
			Qo[2][2] = Q[2][id2];
			
			return 0;
		}
		
		if (nIter < 4)
			thresh = 0.2 * so / SQR(n);
		else
			thresh = 0.0;
		
		// Do sweep
		for (int p=0; p < n; p++)
			for (int q=p+1; q < n; q++)
			{
				g = 100.0 * fabs(A[p][q]);
				if (nIter > 4  &&  fabs(w[p]) + g == fabs(w[p])
					&&  fabs(w[q]) + g == fabs(w[q]))
				{
					A[p][q] = 0.0;
				}
				else if (fabs(A[p][q]) > thresh)
				{
					// Calculate Jacobi transformation
					h = w[q] - w[p];
					if (fabs(h) + g == fabs(h))
					{
						t = A[p][q] / h;
					}
					else
					{
						theta = 0.5 * h / A[p][q];
						if (theta < 0.0)
							t = -1.0 / (sqrt(1.0 + SQR(theta)) - theta);
						else
							t = 1.0 / (sqrt(1.0 + SQR(theta)) + theta);
					}
					c = 1.0/sqrt(1.0 + SQR(t));
					s = t * c;
					z = t * A[p][q];
					
					// Apply Jacobi transformation
					A[p][q] = 0.0;
					w[p] -= z;
					w[q] += z;
					for (int r=0; r < p; r++)
					{
						t = A[r][p];
						A[r][p] = c*t - s*A[r][q];
						A[r][q] = s*t + c*A[r][q];
					}
					for (int r=p+1; r < q; r++)
					{
						t = A[p][r];
						A[p][r] = c*t - s*A[r][q];
						A[r][q] = s*t + c*A[r][q];
					}
					for (int r=q+1; r < n; r++)
					{
						t = A[p][r];
						A[p][r] = c*t - s*A[q][r];
						A[q][r] = s*t + c*A[q][r];
					}
					
					// Update eigenvectors
#ifndef EVALS_ONLY          
					for (int r=0; r < n; r++)
					{
						t = Q[r][p];
						Q[r][p] = c*t - s*Q[r][q];
						Q[r][q] = s*t + c*Q[r][q];
					}
#endif
				}
			}
	}
	
	// Ordering eigenvalues (buble sort)
	id0 = 0;
	id1 = 1;
	id2 = 2;
	if ( fabs(w[1]) > fabs(w[0]) ) {
		hold = w[1];
		w[1] = w[0];
		w[0] = hold;
		id0 = 1;
		id1 = 0;
	}
	if ( fabs(w[2]) > fabs(w[1]) ) {
		hold = w[2];
		w[2] = w[1];
		w[1] = hold;
		hold = id1;
		id1 = 2;
		id2 = id1;
	}
	if ( fabs(w[1]) > fabs(w[0]) ) {
		hold = w[1];
		w[1] = w[0];
		w[0] = hold;
		hold = id0;
		id0 = id1;
		id1 = hold;
	}
	
	// Reorder eigenvectors
	Qo[0][0] = Q[0][id0];	
	Qo[1][0] = Q[1][id0];
	Qo[2][0] = Q[2][id0];	
	Qo[0][1] = Q[0][id1];	
	Qo[1][1] = Q[1][id1];
	Qo[2][1] = Q[2][id1];		
	Qo[0][2] = Q[0][id2];	
	Qo[1][2] = Q[1][id2];
	Qo[2][2] = Q[2][id2];	
	
	return -1;
}

