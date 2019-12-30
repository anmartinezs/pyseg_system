/*
 *  desyevvmex.cpp
 
 Stub for calculating the eigenvectors (normalized and ordered) and eigenvalues (ordered) from 
 directional derivatives (reshaped into arrays). Analytical Kopp2008.
 
 *  
 *
 *  Created by Antonio Martínez on 1/2/13.
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
double *Ixx, *Iyy, *Izz, *Ixy, *Ixz, *Iyz;
double *L1, *L2, *L3, *V1x, *V1y, *V1z, *V2x, *V2y, *V2z, *V3x, *V3y, *V3z;
pthread_mutex_t mutex;

int dsyevv3(double A[3][3], double Q[3][3], double w[3]);
int dsyevc3(double A[3][3], double w[3]);
void* desyevv3stub( void* ptr );

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{ 
	mwSize m;
	int nta, nth;
	pthread_t* threads;
	size_t len, len64;
	long long int dat64;	
	
	// Check for proper number of arguments
    if (nrhs != 6) { 
		mexErrMsgTxt("desyevvmex.cpp: Six input arguments required."); 
		return;
    } else if (nlhs > 12) {
		mexErrMsgTxt("desyevvmex.cpp: Too many output arguments."); 
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
		mexErrMsgTxt("desyevvmex.cpp: No Cache L2 detected.");
		return;
	}
	task_size = ceil( float(CACHE_L2_USED*dat64) / 18 );
	
	// Check the dimensions of the input arrays
    m = mxGetM(prhs[0]);
	ld = m;
	if ( (m!=mxGetM(prhs[1])) || (m!=mxGetM(prhs[2])) || (m!=mxGetM(prhs[3])) || 
		(m!=mxGetM(prhs[4])) || (m!=mxGetM(prhs[5])) ) {
		mexErrMsgTxt("desyevvmex.cpp: Dimensions mismatch."); 
		return;
	}
    if ( !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) || 
		!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) ||
		!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) ||
		!mxIsDouble(prhs[3]) || mxIsComplex(prhs[3]) ||
		!mxIsDouble(prhs[4]) || mxIsComplex(prhs[4]) ||
		!mxIsDouble(prhs[5]) || mxIsComplex(prhs[5]) ){ 
			mexErrMsgTxt("desyevvmex.cpp: Requires double real data."); 
			return;
	} 
	
	// Create the arrays for holding the output results
    plhs[0] = mxCreateDoubleMatrix( m, 1, mxREAL );
	plhs[1] = mxCreateDoubleMatrix( m, 1, mxREAL );
	plhs[2] = mxCreateDoubleMatrix( m, 1, mxREAL );
	plhs[3] = mxCreateDoubleMatrix( m, 1, mxREAL );
	plhs[4] = mxCreateDoubleMatrix( m, 1, mxREAL );
    plhs[5] = mxCreateDoubleMatrix( m, 1, mxREAL );	
    plhs[6] = mxCreateDoubleMatrix( m, 1, mxREAL );
	plhs[7] = mxCreateDoubleMatrix( m, 1, mxREAL );
	plhs[8] = mxCreateDoubleMatrix( m, 1, mxREAL );
	plhs[9] = mxCreateDoubleMatrix( m, 1, mxREAL );
	plhs[10] = mxCreateDoubleMatrix( m, 1, mxREAL );
    plhs[11] = mxCreateDoubleMatrix( m, 1, mxREAL );
	
	// Assign pointers to data
	Ixx = mxGetPr( prhs[0] );
	Iyy = mxGetPr( prhs[1] );
	Izz = mxGetPr( prhs[2] );	
	Ixy = mxGetPr( prhs[3] );
	Ixz = mxGetPr( prhs[4] );
	Iyz = mxGetPr( prhs[5] );
	L1 = mxGetPr( plhs[0] );
	L2 = mxGetPr( plhs[1] );
	L3 = mxGetPr( plhs[2] );	
	V1x = mxGetPr( plhs[3] );
	V1y = mxGetPr( plhs[4] );
	V1z = mxGetPr( plhs[5] );
	V2x = mxGetPr( plhs[6] );
	V2y = mxGetPr( plhs[7] );
	V2z = mxGetPr( plhs[8] );
	V3x = mxGetPr( plhs[9] );
	V3y = mxGetPr( plhs[10] );
	V3z = mxGetPr( plhs[11] );
	
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
		mexErrMsgTxt("desyevvmex.cpp: Error creating the mutex.");
		return;
	}
	threads = (pthread_t*)malloc( nth*sizeof(pthread_t) );
	for (int i=0; i<nth; i++) {
		// Update process queue pointers
		if (pthread_create(&threads[i],NULL,&desyevv3stub,NULL)) {
			mexErrMsgTxt("desyevvmex.cpp: Error creating a thread.");
			return;
		}
	}
	
	// Wait for all workers
	for (int i=0; i<nth; i++) {
		if (pthread_join(threads[i],NULL)) {
			mexErrMsgTxt("desyevvmex.cpp: Error waiting the thread termination.");
			return;
		}
	} 
	
	return;
}	

// Thread for calling the function for doing the calculations
void* desyevv3stub( void* ptr ){
	
	int start, end;
	double A[3][3];
	double Q[3][3];
	double w[3];
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
			dsyevv3( A, Q, w );
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
int dsyevv3(double A[3][3], double Q[3][3], double w[3])
// ----------------------------------------------------------------------------
// Calculates the eigenvalues and normalized eigenvectors of a symmetric 3x3
// matrix A using Cardano's method for the eigenvalues and an analytical
// method based on vector cross products for the eigenvectors.
// Only the diagonal and upper triangular parts of A need to contain meaningful
// values. However, all of A may be used as temporary storage and may hence be
// destroyed.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The symmetric input matrix
//   Qo: Storage buffer for eigenvectors
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error
// ----------------------------------------------------------------------------
// Dependencies:
//   dsyevc3()
// ----------------------------------------------------------------------------
// Version history:
//   v1.1 (12 Mar 2012): Removed access to lower triangualr part of A
//     (according to the documentation, only the upper triangular part needs
//     to be filled)
//   v1.0: First released version
// ----------------------------------------------------------------------------
{
#ifndef EVALS_ONLY
	double norm;          // Squared norm or inverse norm of current eigenvector
	double n0, n1;        // Norm of first and second columns of A
	double n0tmp, n1tmp;  // "Templates" for the calculation of n0/n1 - saves a few FLOPS
	double thresh;        // Small number used as threshold for floating point comparisons
	double error;         // Estimated maximum roundoff error in some steps
	double wmax;          // The eigenvalue of maximum modulus
	double f, t;          // Intermediate storage
	int i, j;             // Loop counters
	double hold;
#endif
	
	// Calculate eigenvalues
	dsyevc3(A, w);
	
	// Ordering eigenvalues (buble sort)
	if ( fabs(w[1]) > fabs(w[0]) ) {
		hold = w[1];
		w[1] = w[0];
		w[0] = hold;
	}
	if ( fabs(w[2]) > fabs(w[1]) ) {
		hold = w[2];
		w[2] = w[1];
		w[1] = hold;
	}
	if ( fabs(w[1]) > fabs(w[0]) ) {
		hold = w[1];
		w[1] = w[0];
		w[0] = hold;
	}
	
#ifndef EVALS_ONLY
	wmax = fabs(w[0]);
	if ((t=fabs(w[1])) > wmax)
		wmax = t;
	if ((t=fabs(w[2])) > wmax)
		wmax = t;
	thresh = SQR(8.0 * DBL_EPSILON * wmax);
	
	// Prepare calculation of eigenvectors
	n0tmp   = SQR(A[0][1]) + SQR(A[0][2]);
	n1tmp   = SQR(A[0][1]) + SQR(A[1][2]);
	Q[0][1] = A[0][1]*A[1][2] - A[0][2]*A[1][1];
	Q[1][1] = A[0][2]*A[0][1] - A[1][2]*A[0][0];
	Q[2][1] = SQR(A[0][1]);
	
	// Calculate first eigenvector by the formula
	//   v[0] = (A - w[0]).e1 x (A - w[0]).e2
	A[0][0] -= w[0];
	A[1][1] -= w[0];
	Q[0][0] = Q[0][1] + A[0][2]*w[0];
	Q[1][0] = Q[1][1] + A[1][2]*w[0];
	Q[2][0] = A[0][0]*A[1][1] - Q[2][1];
	norm    = SQR(Q[0][0]) + SQR(Q[1][0]) + SQR(Q[2][0]);
	n0      = n0tmp + SQR(A[0][0]);
	n1      = n1tmp + SQR(A[1][1]);
	error   = n0 * n1;
	
	if (n0 <= thresh)         // If the first column is zero, then (1,0,0) is an eigenvector
	{
		Q[0][0] = 1.0;
		Q[1][0] = 0.0;
		Q[2][0] = 0.0;
	}
	else if (n1 <= thresh)    // If the second column is zero, then (0,1,0) is an eigenvector
	{
		Q[0][0] = 0.0;
		Q[1][0] = 1.0;
		Q[2][0] = 0.0;
	}
	else if (norm < SQR(64.0 * DBL_EPSILON) * error)
	{                         // If angle between A[0] and A[1] is too small, don't use
		t = SQR(A[0][1]);       // cross product, but calculate v ~ (1, -A0/A1, 0)
		f = -A[0][0] / A[0][1];
		if (SQR(A[1][1]) > t)
		{
			t = SQR(A[1][1]);
			f = -A[0][1] / A[1][1];
		}
		if (SQR(A[1][2]) > t)
			f = -A[0][2] / A[1][2];
		norm    = 1.0/sqrt(1 + SQR(f));
		Q[0][0] = norm;
		Q[1][0] = f * norm;
		Q[2][0] = 0.0;
	}
	else                      // This is the standard branch
	{
		norm = sqrt(1.0 / norm);
		for (j=0; j < 3; j++)
			Q[j][0] = Q[j][0] * norm;
	}
	
	
	// Prepare calculation of second eigenvector
	t = w[0] - w[1];
	if (fabs(t) > 8.0 * DBL_EPSILON * wmax)
	{
		// For non-degenerate eigenvalue, calculate second eigenvector by the formula
		//   v[1] = (A - w[1]).e1 x (A - w[1]).e2
		A[0][0] += t;
		A[1][1] += t;
		Q[0][1]  = Q[0][1] + A[0][2]*w[1];
		Q[1][1]  = Q[1][1] + A[1][2]*w[1];
		Q[2][1]  = A[0][0]*A[1][1] - Q[2][1];
		norm     = SQR(Q[0][1]) + SQR(Q[1][1]) + SQR(Q[2][1]);
		n0       = n0tmp + SQR(A[0][0]);
		n1       = n1tmp + SQR(A[1][1]);
		error    = n0 * n1;
		
		if (n0 <= thresh)       // If the first column is zero, then (1,0,0) is an eigenvector
		{
			Q[0][1] = 1.0;
			Q[1][1] = 0.0;
			Q[2][1] = 0.0;
		}
		else if (n1 <= thresh)  // If the second column is zero, then (0,1,0) is an eigenvector
		{
			Q[0][1] = 0.0;
			Q[1][1] = 1.0;
			Q[2][1] = 0.0;
		}
		else if (norm < SQR(64.0 * DBL_EPSILON) * error)
		{                       // If angle between A[0] and A[1] is too small, don't use
			t = SQR(A[0][1]);     // cross product, but calculate v ~ (1, -A0/A1, 0)
			f = -A[0][0] / A[0][1];
			if (SQR(A[1][1]) > t)
			{
				t = SQR(A[1][1]);
				f = -A[0][1] / A[1][1];
			}
			if (SQR(A[1][2]) > t)
				f = -A[0][2] / A[1][2];
			norm    = 1.0/sqrt(1 + SQR(f));
			Q[0][1] = norm;
			Q[1][1] = f * norm;
			Q[2][1] = 0.0;
		}
		else
		{
			norm = sqrt(1.0 / norm);
			for (j=0; j < 3; j++)
				Q[j][1] = Q[j][1] * norm;
		}
	}
	else
	{
		// For degenerate eigenvalue, calculate second eigenvector according to
		//   v[1] = v[0] x (A - w[1]).e[i]
		//   
		// This would really get to complicated if we could not assume all of A to
		// contain meaningful values.
		A[1][0]  = A[0][1];
		A[2][0]  = A[0][2];
		A[2][1]  = A[1][2];
		A[0][0] += w[0];
		A[1][1] += w[0];
		for (i=0; i < 3; i++)
		{
			A[i][i] -= w[1];
			n0       = SQR(A[0][i]) + SQR(A[1][i]) + SQR(A[2][i]);
			if (n0 > thresh)
			{
				Q[0][1]  = Q[1][0]*A[2][i] - Q[2][0]*A[1][i];
				Q[1][1]  = Q[2][0]*A[0][i] - Q[0][0]*A[2][i];
				Q[2][1]  = Q[0][0]*A[1][i] - Q[1][0]*A[0][i];
				norm     = SQR(Q[0][1]) + SQR(Q[1][1]) + SQR(Q[2][1]);
				if (norm > SQR(256.0 * DBL_EPSILON) * n0) // Accept cross product only if the angle between
				{                                         // the two vectors was not too small
					norm = sqrt(1.0 / norm);
					for (j=0; j < 3; j++)
						Q[j][1] = Q[j][1] * norm;
					break;
				}
			}
		}
		
		if (i == 3)    // This means that any vector orthogonal to v[0] is an EV.
		{
			for (j=0; j < 3; j++)
				if (Q[j][0] != 0.0)                                   // Find nonzero element of v[0] ...
				{                                                     // ... and swap it with the next one
					norm          = 1.0 / sqrt(SQR(Q[j][0]) + SQR(Q[(j+1)%3][0]));
					Q[j][1]       = Q[(j+1)%3][0] * norm;
					Q[(j+1)%3][1] = -Q[j][0] * norm;
					Q[(j+2)%3][1] = 0.0;
					break;
				}
		}
	}
	
	
	// Calculate third eigenvector according to
	//   v[2] = v[0] x v[1]
	Q[0][2] = Q[1][0]*Q[2][1] - Q[2][0]*Q[1][1];
	Q[1][2] = Q[2][0]*Q[0][1] - Q[0][0]*Q[2][1];
	Q[2][2] = Q[0][0]*Q[1][1] - Q[1][0]*Q[0][1];
#endif
	
	return 0;
}

// ----------------------------------------------------------------------------
int dsyevc3(double A[3][3], double w[3])
// ----------------------------------------------------------------------------
// Calculates the eigenvalues of a symmetric 3x3 matrix A using Cardano's
// analytical algorithm.
// Only the diagonal and upper triangular parts of A are accessed. The access
// is read-only.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The symmetric input matrix
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error
// ----------------------------------------------------------------------------
{
	double m, c1, c0;
	
	// Determine coefficients of characteristic poynomial. We write
	//       | a   d   f  |
	//  A =  | d*  b   e  |
	//       | f*  e*  c  |
	double de = A[0][1] * A[1][2];                                    // d * e
	double dd = SQR(A[0][1]);                                         // d^2
	double ee = SQR(A[1][2]);                                         // e^2
	double ff = SQR(A[0][2]);                                         // f^2
	m  = A[0][0] + A[1][1] + A[2][2];
	c1 = (A[0][0]*A[1][1] + A[0][0]*A[2][2] + A[1][1]*A[2][2])        // a*b + a*c + b*c - d^2 - e^2 - f^2
	- (dd + ee + ff);
	c0 = A[2][2]*dd + A[0][0]*ee + A[1][1]*ff - A[0][0]*A[1][1]*A[2][2]
	- 2.0 * A[0][2]*de;                                     // c*d^2 + a*e^2 + b*f^2 - a*b*c - 2*f*d*e)
	
	double p, sqrt_p, q, c, s, phi;
	p = SQR(m) - 3.0*c1;
	q = m*(p - (3.0/2.0)*c1) - (27.0/2.0)*c0;
	sqrt_p = sqrt(fabs(p));
	
	phi = 27.0 * ( 0.25*SQR(c1)*(p - c1) + c0*(q + 27.0/4.0*c0));
	phi = (1.0/3.0) * atan2(sqrt(fabs(phi)), q);
	
	c = sqrt_p*cos(phi);
	s = (1.0/M_SQRT3)*sqrt_p*sin(phi);
	
	w[1]  = (1.0/3.0)*(m - c);
	w[2]  = w[1] + s;
	w[0]  = w[1] + c;
	w[1] -= s;
	
	return 0;
}