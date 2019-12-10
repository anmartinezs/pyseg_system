/*
 *  desyevhmex.cpp
 
	Stub for calculating the eigenvectors and eigenvalues from directional derivatives (reshaped
	into arrays). Hybrid Kopp2008.
 
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
#include <unistd.h>
#include <string.h>

// Constants
#define M_SQRT3    1.73205080756887729352744634151   // sqrt(3)
#define CACHE_USED .33
#define BUFFER_SIZE 1024
#define BYTE_PER_KBYTE 1024
#define ERROR_FILE_READ -1
#define ERROR_CACHE_NOT_FOUND -2
#define ERROR_MEM_NOT_FOUND -3

// Macros
#define SQR(x)      ((x)*(x)) 

int sq, lq, ld, task_size;
double *Ixx, *Iyy, *Izz, *Ixy, *Ixz, *Iyz;
double *L1, *L2, *L3, *V1x, *V1y, *V1z, *V2x, *V2y, *V2z, *V3x, *V3y, *V3z;
pthread_mutex_t mutex;

int dsyevh3(double A[3][3], double Q[3][3], double w[3]);
int dsyevc3(double A[3][3], double w[3]);
inline void dsytrd3(double A[3][3], double Q[3][3], double d[3], double e[2]);
int dsyevq3(double A[3][3], double Q[3][3], double w[3]);
void* desyevh3stub( void* ptr );
long long int get_cache_size();

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{ 
	mwSize m;
	int nta, nth, num_threads;
	pthread_t* threads;
	long long int dat64;
	
	// Check for proper number of arguments
    if (nrhs != 6) { 
		mexErrMsgTxt("desyevhmex.cpp: Six input arguments required."); 
		return;
    } else if (nlhs > 12) {
		mexErrMsgTxt("desyevhmex.cpp: Too many output arguments."); 
		return;
    }
	
	// Get computer information to set the number of thread and the size of buffers
	num_threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
	if (num_threads<1) {
		mexErrMsgTxt("desyevhmex_single.cpp: No active cpu detected.");
		return;
	}
	dat64 = get_cache_size();
	if (dat64<1) {
		mexErrMsgTxt("desyevhmex_single.cpp: No Cache detected.");
		return;
	}
	task_size = ceil( float(CACHE_USED*dat64) / 18 );
	
	// Check the dimensions of the input arrays
    m = mxGetM(prhs[0]);
	ld = m;
	if ( (m!=mxGetM(prhs[1])) || (m!=mxGetM(prhs[2])) || (m!=mxGetM(prhs[3])) || 
		(m!=mxGetM(prhs[4])) || (m!=mxGetM(prhs[5])) ) {
		mexErrMsgTxt("desyevhmex.cpp: Dimensions mismatch."); 
		return;
	}
    if ( !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) || 
		!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) ||
		!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) ||
		!mxIsDouble(prhs[3]) || mxIsComplex(prhs[3]) ||
		!mxIsDouble(prhs[4]) || mxIsComplex(prhs[4]) ||
		!mxIsDouble(prhs[5]) || mxIsComplex(prhs[5]) ){ 
		mexErrMsgTxt("desyevhmex.cpp: Requires double real data."); 
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
		mexErrMsgTxt("desyevhmex.cpp: Error creating the mutex.");
		return;
	}
	threads = (pthread_t*)malloc( nth*sizeof(pthread_t) );
	for (int i=0; i<nth; i++) {
		// Update process queue pointers
		if (pthread_create(&threads[i],NULL,&desyevh3stub,NULL)) {
			mexErrMsgTxt("desyevhmex.cpp: Error creating a thread.");
			return;
		}
	}
	
	// Wait for all workers
	for (int i=0; i<nth; i++) {
		if (pthread_join(threads[i],NULL)) {
			mexErrMsgTxt("desyevhmex.cpp: Error waiting the thread termination.");
			return;
		}
	} 
	
	return;
}	

// Thread for calling the function for doing the calculations
void* desyevh3stub( void* ptr ){
	
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
			dsyevh3( A, Q, w );
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

	return NULL;
}

// ----------------------------------------------------------------------------
int dsyevh3(double A[3][3], double Q[3][3], double w[3])
// ----------------------------------------------------------------------------
// Calculates the eigenvalues and normalized eigenvectors of a symmetric 3x3
// matrix A using Cardano's method for the eigenvalues and an analytical
// method based on vector cross products for the eigenvectors. However,
// if conditions are such that a large error in the results is to be
// expected, the routine falls back to using the slower, but more
// accurate QL algorithm. Only the diagonal and upper triangular parts of A need
// to contain meaningful values. Access to A is read-only.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The symmetric input matrix
//   Q: Storage buffer for eigenvectors
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error
// ----------------------------------------------------------------------------
// Dependencies:
//   dsyevc3(), dsytrd3(), dsyevq3()
// ----------------------------------------------------------------------------
// Version history:
//   v1.1: Simplified fallback condition --> speed-up
//   v1.0: First released version
// ----------------------------------------------------------------------------
{
#ifndef EVALS_ONLY
	double norm;          // Squared norm or inverse norm of current eigenvector
	//  double n0, n1;        // Norm of first and second columns of A
	double error;         // Estimated maximum roundoff error
	double t, u;          // Intermediate storage
	double hold;
	int j;                // Loop counter
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
	//  n0 = SQR(A[0][0]) + SQR(A[0][1]) + SQR(A[0][2]);
	//  n1 = SQR(A[0][1]) + SQR(A[1][1]) + SQR(A[1][2]);
	
	t = fabs(w[0]);
	if ((u=fabs(w[1])) > t)
		t = u;
	if ((u=fabs(w[2])) > t)
		t = u;
	if (t < 1.0)
		u = t;
	else
		u = SQR(t);
	error = 256.0 * DBL_EPSILON * SQR(u);
	//  error = 256.0 * DBL_EPSILON * (n0 + u) * (n1 + u);
	
	Q[0][1] = A[0][1]*A[1][2] - A[0][2]*A[1][1];
	Q[1][1] = A[0][2]*A[0][1] - A[1][2]*A[0][0];
	Q[2][1] = SQR(A[0][1]);
	
	// Calculate first eigenvector by the formula
	//   v[0] = (A - w[0]).e1 x (A - w[0]).e2
	Q[0][0] = Q[0][1] + A[0][2]*w[0];
	Q[1][0] = Q[1][1] + A[1][2]*w[0];
	Q[2][0] = (A[0][0] - w[0]) * (A[1][1] - w[0]) - Q[2][1];
	norm    = SQR(Q[0][0]) + SQR(Q[1][0]) + SQR(Q[2][0]);
	
	// If vectors are nearly linearly dependent, or if there might have
	// been large cancellations in the calculation of A[i][i] - w[0], fall
	// back to QL algorithm
	// Note that this simultaneously ensures that multiple eigenvalues do
	// not cause problems: If w[0] = w[1], then A - w[0] * I has rank 1,
	// i.e. all columns of A - w[0] * I are linearly dependent.
	if (norm <= error)
		return dsyevq3(A, Q, w);
	else                      // This is the standard branch
	{
		norm = sqrt(1.0 / norm);
		for (j=0; j < 3; j++)
			Q[j][0] = Q[j][0] * norm;
	}
	
	// Calculate second eigenvector by the formula
	//   v[1] = (A - w[1]).e1 x (A - w[1]).e2
	Q[0][1]  = Q[0][1] + A[0][2]*w[1];
	Q[1][1]  = Q[1][1] + A[1][2]*w[1];
	Q[2][1]  = (A[0][0] - w[1]) * (A[1][1] - w[1]) - Q[2][1];
	norm     = SQR(Q[0][1]) + SQR(Q[1][1]) + SQR(Q[2][1]);
	if (norm <= error)
		return dsyevq3(A, Q, w);
	else
	{
		norm = sqrt(1.0 / norm);
		for (j=0; j < 3; j++)
			Q[j][1] = Q[j][1] * norm;
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

// ----------------------------------------------------------------------------
inline void dsytrd3(double A[3][3], double Q[3][3], double d[3], double e[2])
// ----------------------------------------------------------------------------
// Reduces a symmetric 3x3 matrix to tridiagonal form by applying
// (unitary) Householder transformations:
//            [ d[0]  e[0]       ]
//    A = Q . [ e[0]  d[1]  e[1] ] . Q^T
//            [       e[1]  d[2] ]
// The function accesses only the diagonal and upper triangular parts of
// A. The access is read-only.
// ---------------------------------------------------------------------------
{
	const int n = 3;
	double u[n], q[n];
	double omega, f;
	double K, h, g;
	
	// Initialize Q to the identitity matrix
#ifndef EVALS_ONLY
	for (int i=0; i < n; i++)
	{
		Q[i][i] = 1.0;
		for (int j=0; j < i; j++)
			Q[i][j] = Q[j][i] = 0.0;
	}
#endif
	
	// Bring first row and column to the desired form 
	h = SQR(A[0][1]) + SQR(A[0][2]);
	if (A[0][1] > 0)
		g = -sqrt(h);
	else
		g = sqrt(h);
	e[0] = g;
	f    = g * A[0][1];
	u[1] = A[0][1] - g;
	u[2] = A[0][2];
	
	omega = h - f;
	if (omega > 0.0)
	{
		omega = 1.0 / omega;
		K     = 0.0;
		for (int i=1; i < n; i++)
		{
			f    = A[1][i] * u[1] + A[i][2] * u[2];
			q[i] = omega * f;                  // p
			K   += u[i] * f;                   // u* A u
		}
		K *= 0.5 * SQR(omega);
		
		for (int i=1; i < n; i++)
			q[i] = q[i] - K * u[i];
		
		d[0] = A[0][0];
		d[1] = A[1][1] - 2.0*q[1]*u[1];
		d[2] = A[2][2] - 2.0*q[2]*u[2];
		
		// Store inverse Householder transformation in Q
#ifndef EVALS_ONLY
		for (int j=1; j < n; j++)
		{
			f = omega * u[j];
			for (int i=1; i < n; i++)
				Q[i][j] = Q[i][j] - f*u[i];
		}
#endif
		
		// Calculate updated A[1][2] and store it in e[1]
		e[1] = A[1][2] - q[1]*u[2] - u[1]*q[2];
	}
	else
	{
		for (int i=0; i < n; i++)
			d[i] = A[i][i];
		e[1] = A[1][2];
	}
}

// ----------------------------------------------------------------------------
int dsyevq3(double A[3][3], double Q[3][3], double w[3])
// ----------------------------------------------------------------------------
// Calculates the eigenvalues and normalized eigenvectors of a symmetric 3x3
// matrix A using the QL algorithm with implicit shifts, preceded by a
// Householder reduction to tridiagonal form.
// The function accesses only the diagonal and upper triangular parts of A.
// The access is read-only.
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
// Dependencies:
//   dsytrd3()
// ----------------------------------------------------------------------------
{
	const int n = 3;
	double e[3];                   // The third element is used only as temporary workspace
	double g, r, p, f, b, s, c, t; // Intermediate storage
	int nIter;
	int m;
	
	// Transform A to real tridiagonal form by the Householder method
	dsytrd3(A, Q, w, e);
	
	// Calculate eigensystem of the remaining real symmetric tridiagonal matrix
	// with the QL method
	//
	// Loop over all off-diagonal elements
	for (int l=0; l < n-1; l++)
	{
		nIter = 0;
		while (1)
		{
			// Check for convergence and exit iteration loop if off-diagonal
			// element e(l) is zero
			for (m=l; m <= n-2; m++)
			{
				g = fabs(w[m])+fabs(w[m+1]);
				if (fabs(e[m]) + g == g)
					break;
			}
			if (m == l)
				break;
			
			if (nIter++ >= 30)
				return -1;
			
			// Calculate g = d_m - k
			g = (w[l+1] - w[l]) / (e[l] + e[l]);
			r = sqrt(SQR(g) + 1.0);
			if (g > 0)
				g = w[m] - w[l] + e[l]/(g + r);
			else
				g = w[m] - w[l] + e[l]/(g - r);
			
			s = c = 1.0;
			p = 0.0;
			for (int i=m-1; i >= l; i--)
			{
				f = s * e[i];
				b = c * e[i];
				if (fabs(f) > fabs(g))
				{
					c      = g / f;
					r      = sqrt(SQR(c) + 1.0);
					e[i+1] = f * r;
					c     *= (s = 1.0/r);
				}
				else
				{
					s      = f / g;
					r      = sqrt(SQR(s) + 1.0);
					e[i+1] = g * r;
					s     *= (c = 1.0/r);
				}
				
				g = w[i+1] - p;
				r = (w[i] - g)*s + 2.0*c*b;
				p = s * r;
				w[i+1] = g + p;
				g = c*r - b;
				
				// Form eigenvectors
#ifndef EVALS_ONLY
				for (int k=0; k < n; k++)
				{
					t = Q[k][i+1];
					Q[k][i+1] = s*Q[k][i] + c*t;
					Q[k][i]   = c*Q[k][i] - s*t;
				}
#endif 
			}
			w[l] -= p;
			e[l]  = g;
			e[m]  = 0.0;
		}
	}
	
	return 0;
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
