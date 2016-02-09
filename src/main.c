#include <stdio.h>

/************************* meschach start *************************/
#include "matrix.h"
#include "matrix2.h"
/* license:
 * 		- copyrighted
 * description: 
 *		- linear combinations of vectors and matrices; inner products;
 *			matrix-vector and matrix-matrix products
 *		- solving linear systems of equations
 *		- solving least-squares problems
 *		- computing eigenvalues (or singular values)
 *		- finding "nice" bases for subspaces like ranges of matrices and null spaces.
 * installation:
 * 		- basic configure, make, make test/check
 * 		- configure script broken
 * 		- contains prebuilt platform specific makefiles
 * use:
 * 		- link to meschach.a and math library
 * 		- flags: -lmeschach -lm
 * documentation:
 * 		- http://homepage.math.uiowa.edu/~dstewart/meschach/meschach.html
 * 		- code not well commented
 * 		- some documentation on website, remainder in book (available as pdf)
 * comments:
 * 		- pronunciation "me-shark"
 * 		- fast
 * 		- fairly small
 * 		- seems easy to include additional functionality if needed (add .h file)
 * 		- user friendly
 * 		- benefits of lapack -> data structures for matrices, easy to resize, 
 * 		- sparse matrix support
 * ARM compatible
 *		- ftp://ftp.netbsd.org/pub/pkgsrc/current/pkgsrc/math/meschach/README.html
 *		- available, not sure how optimized it is
 */ 	
/************************* meschach end *************************/



/************************* CBLAS start *************************/
#include <cblas.h>
/* license:
 * 		- copyrighted, can be included in commercial software packages
 * description: 
 *		- C interface to BLAS
 *		- low level vector and matrix operations
 *		- BLAS is the standard used for many other linear algebra libraries
 * installation:
 * 		- easy install -> blas and cblas through package manager
 * use:
 * 		- flags: -lcblas
 * documentation:
 * 		- http://www.netlib.org/blas/
 * 		- http://techpubs.sgi.com/library/tpl/cgi-bin/getdoc.cgi?cmd=getdoc&coll=0650&db=man&fname=3%20INTRO_CBLAS
 * 		- https://developer.apple.com/library/mac/documentation/Accelerate/Reference/BLAS_Ref/#//apple_ref/c/func/cblas_sgemm
 * comments:
 * 		- took some digging to find function descriptions, best was on apple website
 * 		- functions require many inputs and are not very clear
 * 		- probably doesn't have enough functionality to use on its own
 * ARM compatible
 *		- ???
 */ 	
/************************* CBLAS end *************************/



/************************* LAPACKE start *************************/
#include <lapacke.h>
/* license:
 * 		- copyrighted 
 * description: 
 *		- provides a c standard interface to lapack (fortran)
 *		- provides routines for solving systems of simultaneous linear equations, 
 *			least-squares solutions of linear systems of equations, eigenvalue problems, 
 *			and singular value problems. The associated matrix factorizations 
 *			(LU, Cholesky, QR, SVD, Schur, generalized Schur) are also provided
 * installation:
 * 		- installed with package manager
 * use:
 * 		- flags: -llapacke
 * documentation: 
 * 		- http://www.netlib.org/lapack/explore-html/d7/d7c/example__user_8c.html
 * 		- http://www.netlib.org/lapack/lapacke.html
 * comments:
 * 		- specific way to define matrices for functions, or use cast
 * 		- this library has been around for a long time (lapack)
 * 		- seems to have good amount of functionality
 * 		- no data structures or support for creating matrix
 * ARM compatible
 *		- http://ds.arm.com/solutions/high-performance-computing/arm-performance-libraries/
 *		- above link has BLAS LAPACK and FFT support (not free)
 *		- probably other options as well
 */ 	
/************************* LAPACKE end *************************/



/************************* GSL start *************************/
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
/* license:
 * 		- GNU General Public License
 * description: 
 *		- The library provides a wide range of mathematical routines such as random number generators,
 *			special functions and least-squares fitting. There are over 1000 functions in total
 * installation:
 * 		- package manager
 * use:
 * 		- flags: -lgsl -lgslcblas
 * documentation: 
 * 		- http://www.gnu.org/software/gsl/manual/html_node/ 
 * comments:
 * 		- GSL requires a BLAS library for vector and matrix operations.
 * 		- The default CBLAS library supplied with GSL can be replaced by the tuned ATLAS library for better performance
 * 		- has a lot of uneeded funtionality
 * ARM compatible
 *		- can be cross compiled (optimization? effeciency?)
 */ 	
/************************* GSL end *************************/



/*
 * Other possibilities and info:
 *		- List of open source linear algebra libraries http://www.netlib.org/utk/people/JackDongarra/la-sw.html
 *		- BLAS wikipedia page lists libraries that use BLAS
 * 		- ATLAS http://math-atlas.sourceforge.net/
 * 			- optimized (per machine) BLAS library with some LAPACK functionality
 * 		- OpenBLAS
 * 			- optimized BLAS library (arm support)
 *		- CLAPACK http://www.netlib.org/clapack/
 *			- c version of lapack - not standardized
 *		- LAPACK http://www.netlib.org/lapack/
 *		- NAG http://www.nag.com/numeric/numerical_libraries.asp
 *		- MCSDK HPC 3.x
 *			- http://processors.wiki.ti.com/index.php/MCSDK_HPC_3.x_Linear_Algebra_Library
 */

// print matrix for lapacke
void print_matrix_rowmajor( char* desc, lapack_int m, lapack_int n, double* mat, lapack_int ldm );

int main(void)
{
	int i,j;

	/************************* meschach start *************************/
	// creating matrix
	MAT *A_mesch=NULL, *B_mesch=NULL, *C_mesch=NULL;
	A_mesch = m_get(3,3); // 3x3 matrix
	B_mesch = m_get(3,3);
	
	// loading values
	m_zero(A_mesch); // zeroize
	m_ident(A_mesch); // identity
	
	// printing matrix
	printf("meschach matrix A_mesch\n");	m_output(A_mesch); printf("\n");
	
	// accessing individual elements
	B_mesch->me[0][0] = 1;
	B_mesch->me[0][1] = 3;
	B_mesch->me[0][2] = 3;
	B_mesch->me[1][0] = 1; B_mesch->me[1][1] = 4;
	B_mesch->me[1][2] = 3;
	B_mesch->me[2][0] = 1;
	B_mesch->me[2][1] = 3;
	B_mesch->me[2][2] = 4;
	
	// add
	C_mesch = m_add(A_mesch,B_mesch,MNULL);
	
	// multiply
	C_mesch = m_mlt(A_mesch,B_mesch,MNULL);

	// transpose
	C_mesch = m_transp(B_mesch,MNULL);	

	// inverse
	C_mesch = m_inverse(B_mesch,MNULL);	
	
	// solve Ax=b
	double mesch_data[4][4] = 
	{
		{0.18, 0.60, 0.57, 0.96},
		{0.41, 0.24, 0.99, 0.58},
		{0.14, 0.30, 0.97, 0.66},
		{0.51, 0.13, 0.19, 0.85}
	};
	VEC *mesch_x, *mesch_b;
	MAT	*mesch_A, *LU;
	PERM *pivot;
	mesch_b = v_get(4);
	mesch_b->ve[0] = 1;
	mesch_b->ve[1] = 2;
	mesch_b->ve[2] = 3;
	mesch_b->ve[3] = 4;
	mesch_A = m_get(4,4);
	for(i=0; i<4; i++)
		for(j=0; j<4; j++)
			mesch_A->me[i][j] = mesch_data[i][j];
	LU = m_get(mesch_A->m,mesch_A->n);
	LU = m_copy(mesch_A,LU);
	pivot = px_get(mesch_A->m);
	LUfactor(LU,pivot);
	mesch_x = LUsolve(LU,pivot,mesch_b,VNULL);
	printf("meschach solve Ax=b\n"); v_output(mesch_x); printf("\n");

	// free matrix
	m_free(A_mesch);
	m_free(B_mesch);
	m_free(C_mesch);
	m_free(mesch_A);
	m_free(LU);
	v_free(mesch_b);
	v_free(mesch_x);
	/************************* meschach end *************************/



	/************************* CBLAS start *************************/
	// creating matrix - no special functions
	float A_cblas[3][3] = 
	{
		{1,0,0},
		{0,1,0},
		{0,0,1}
	};
	float B_cblas[3][3] = 
	{
		{1,3,3},
		{1,4,3},
		{1,3,4}
	};
	float C_cblas[3][3];
	
	// multiply
	/* cblas_sgemm ( order , TransA , TransB , M , N , K , alpha , *A , lda , *B , ldb , beta , *C , ldc );
	 * Description: C←αAB + βC
	 * Order: row-major (C) or column-major
	 * TransA: whether to transpose matrix A.
	 * TransB: whether to transpose matrix B.
	 * M: Number of rows in matrices A and C.
	 * N: Number of columns in matrices B and C.
	 * K: Number of columns in matrix A; number of rows in matrix B.
	 * alpha: Scaling factor for the product of matrices A and B.
	 * A: Matrix A.
	 * lda: The size of the first dimention of matrix A; if you are passing a matrix A[m][n], the value should be m.
	 * B: Matrix B.
	 * ldb: The size of the first dimention of matrix B; if you are passing a matrix B[m][n], the value should be m.
	 * beta: Scaling factor for matrix C.
	 * C: Matrix C.
	 * ldc: The size of the first dimention of matrix C; if you are passing a matrix C[m][n], the value should be m.
	 */
	cblas_sgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1, (float *) A_cblas,
										3, (float *) B_cblas, 3, 0, (float *) C_cblas, 3);	
	printf("CBLAS matrix C=A*B\nresult:\n");
	for(i=0; i<3; i++) {
		for(j=0; j<3; j++) 
			printf("%f\t",C_cblas[i][j]);
		printf("\n");
	}
	printf("\n");
	/************************* CBLAS end *************************/

	/************************* LAPACKE start *************************/
	// solve Ax=b
	// variables
	lapack_int n=5, nrhs=3, lda, ldb, info;
	double *A, *b;
	lapack_int *ipiv;

	/* Initialization */
	lda=n, ldb=nrhs;
	A = (double *)malloc(n*n*sizeof(double)) ;
	b = (double *)malloc(n*nrhs*sizeof(double)) ;
	ipiv = (lapack_int *)malloc(n*sizeof(lapack_int)) ;
	for( i = 0; i < n; i++ ) {
		for( j = 0; j < n; j++ ) A[i*lda+j] = ((double) rand()) / ((double) RAND_MAX) - 0.5;
	}
	for(i=0;i<n*nrhs;i++)
	    b[i] = ((double) rand()) / ((double) RAND_MAX) - 0.5;

	printf( "LAPACKE_dgesv solve A*X = B\n" );
	print_matrix_rowmajor( "Entry Matrix A", n, n, A, lda );
	print_matrix_rowmajor( "Right Rand Side b", n, nrhs, b, ldb );
	info = LAPACKE_dgesv( LAPACK_ROW_MAJOR, n, nrhs, A, lda, ipiv,
						b, ldb );
	/* Check for the exact singularity */
	if( info > 0 ) {
		printf( "The diagonal element of the triangular factor of A,\n" );
		printf( "U(%i,%i) is zero, so that A is singular;\n", info, info );
		printf( "the solution could not be computed.\n" );
		exit( 1 );
	}
	if (info <0) exit( 1 );

	/* Print solution */
	print_matrix_rowmajor( "Solution", n, nrhs, b, ldb );
	printf("\n");
	/************************* LAPACKE end *************************/
	
	
	
	/************************* GSL start *************************/
	// create a matrix	
	gsl_matrix * A_GSL = gsl_matrix_alloc (4,4);
	gsl_matrix * B_GSL = gsl_matrix_alloc(4,4);
   
	// load a matrix with values	
	gsl_matrix_set_identity(A_GSL);
	gsl_matrix_set_all(A_GSL, 4.2);
	for (i = 0; i < 4; i++)
		for (j = 0; j < 4; j++) {
			gsl_matrix_set (A_GSL, i, j, 0.23 + 100*i + j);
			gsl_matrix_set (B_GSL, i, j, 0.23 + 10*i);
		}

	// print matrix (get values)
	printf("A_GSL\n");
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++)
			printf ("%g\t",gsl_matrix_get(A_GSL, i, j));
		printf("\n");
	}

	// add matrix (must be same dimensions) results stored in matrix A
	int res = gsl_matrix_add(A_GSL,B_GSL);
	if(res != 0)
		return 1;

	// multiply matrix
	// use cblas (multiply elementwise available though)

	// solve Ax=b
	double a_data[] = { 0.18, 0.60, 0.57, 0.96,
	                      0.41, 0.24, 0.99, 0.58,
	                      0.14, 0.30, 0.97, 0.66,
	                      0.51, 0.13, 0.19, 0.85 };
	double b_data[] = { 1.0, 2.0, 3.0, 4.0 };
	gsl_matrix_view m = gsl_matrix_view_array (a_data, 4, 4);
	gsl_vector_view bb = gsl_vector_view_array (b_data, 4);
	gsl_vector *x = gsl_vector_alloc (4);
	int s;
	gsl_permutation * p = gsl_permutation_alloc (4);
	gsl_linalg_LU_decomp (&m.matrix, p, &s);
	gsl_linalg_LU_solve (&m.matrix, p, &bb.vector, x);
	printf("\n");
	printf ("GSL solve Ax = b\n");
	gsl_vector_fprintf (stdout, x, "%g");
	printf("\n");

	// free matrix
	gsl_matrix_free(A_GSL);
	gsl_matrix_free(B_GSL);
	gsl_permutation_free (p);
	gsl_vector_free (x);
	/************************* GSL end *************************/

	return 0;
}

 /* Auxiliary routine: printing a matrix */
void print_matrix_rowmajor( char* desc, lapack_int m, lapack_int n, double* mat, lapack_int ldm ) {
	lapack_int i, j;
	printf( "%s\n", desc );
	for( i = 0; i < m; i++ ) {
		for( j = 0; j < n; j++ ) printf( " %6.2f", mat[i*ldm+j] );
	printf( "\n" );
	}
}


