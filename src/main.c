#include <stdio.h>

/************************* meschach start *************************/
#include "matrix.h"
#include "matrix2.h"
/* license:
 * 		- free to use but copyrighted
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
 * ARM compatible
 *		- ???
 */ 	
/************************* meschach end *************************/



/************************* CBLAS start *************************/
#include <cblas.h>
/* license:
 * 		- copyrighted, can be included in commercial software packages,
 * 			credit authors, document any changes
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



/*
 * Other possibilities and info:
 *		- List of open source linear algebra libraries http://www.netlib.org/utk/people/JackDongarra/la-sw.html
 *		- BLAS wikipedia page lists libraries that use BLAS
 * 		- ATLAS http://math-atlas.sourceforge.net/
 * 			- optimized (per machine) BLAS library with some LAPACK functionality
 * 		- GSL http://www.gnu.org/software/gsl/
 *		- CLAPACK http://www.netlib.org/clapack/
 *			- c version of lapack - not standardized
 *		- LAPACK http://www.netlib.org/lapack/
 *		- LAPACKE http://www.netlib.org/lapack/lapacke.html
 *			- standardized c version of lapack
 *		- NAG http://www.nag.com/numeric/numerical_libraries.asp
 */

int main(void)
{
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
	B_mesch->me[1][0] = 1;
	B_mesch->me[1][1] = 4;
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

	// free matrix
	m_free(A_mesch);
	m_free(B_mesch);
	m_free(C_mesch);
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
	for(int i=0; i<3; i++) {
		for(int j=0; j<3; j++) 
			printf("%f\t",C_cblas[i][j]);
		printf("\n");
	}
	/************************* CBLAS end *************************/

	return 0;
}
