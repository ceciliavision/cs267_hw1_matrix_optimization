#include <immintrin.h>
#include "string.h"
#include <stdio.h>

const char* dgemm_desc = "Blocked dgemm with unrolling, reuse and padding.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 32
#endif

#define min(a,b) (((a)<(b))?(a):(b))

static void do_block (const int lda, const int M, const int N, const int K, double* restrict A, double* restrict B, double* restrict C);
static void do_block_unroll2(const int lda, const int M, const int N, const int K, double* restrict A, double* restrict B, double* restrict C);
static void do_block_unroll4(const int lda, const int M, const int N, const int K, double* restrict A, double* restrict B, double* restrict C);
static void do_block_unroll8(const int lda, const int M, const int N, const int K, double* restrict A, double* restrict B, double* restrict C);
static void pad(double* restrict padA, double* restrict A, const int lda, const int newlda);
static void unpad(double* restrict padA, double* restrict A, const int lda, const int newlda);
void square_dgemm (const int lda, double* restrict A, double* restrict B, double* restrict C);

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (const int lda, const int M, const int N, const int K, double* restrict A, double* restrict B, double* restrict C)
{
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
  {
    /* For each column j of B */ 
    for (int j = 0; j < N; ++j) 
    {
      /* Compute C(i,j) */
      double cij = C[i+j*lda];
      for (int k = 0; k < K; ++k){
        cij += A[i+k*lda] * B[k+j*lda];
        C[i+j*lda] = cij;
      }
    }
  }
}

/* UNROLL 2: This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block_unroll2 (const int lda, const int M, const int N, const int K, double* restrict A, double* restrict B, double* restrict C)
{
  double cij, Aik, Aik_1;
  /* For each row i of A */

  for (int i = 0; i < M; ++i)
  {
    /* For each column j of B */ 
    for (int k = 0; k < K; k+=2) 
    {
      /* Compute C(i,j) */
      Aik = A[i+k*lda];
      Aik_1 = A[i+(k+1)*lda];
      for (int j = 0; j < N; ++j){
        cij = Aik * B[k+j*lda];
        cij += Aik_1 * B[k+j*lda+1];
        C[i+j*lda] += cij;
      }
    }
  }
}

/* UNROLL 4: This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block_unroll4(const int lda, const int M, const int N, const int K, double* restrict A, double* restrict B, double* restrict C)
{
  double cij, Aik, Aik_1, Aik_2, Aik_3;
  /* For each row i of A */

  for (int i = 0; i < M; ++i){
    /* For each column j of B */ 
    for (int k = 0; k < K; k+=4) 
    {
      /* Compute C(i,j) */
      Aik = A[i+k*lda];
      Aik_1 = A[i+(k+1)*lda];
      Aik_2 = A[i+(k+2)*lda];
      Aik_3 = A[i+(k+3)*lda];
      for (int j = 0; j < N; ++j) {
        cij = Aik * B[k+j*lda];
        cij += Aik_1 * B[k+j*lda+1];
        cij += Aik_2 * B[k+j*lda+2];
        cij += Aik_3 * B[k+j*lda+3];
        C[i+j*lda] += cij;
      }
    }
  }
}



/* UNROLL 8: This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block_unroll8(const int lda, const int M, const int N, const int K, double* restrict A, double* restrict B, double* restrict C)
{
  double cij, Aik, Aik_1, Aik_2, Aik_3, Aik_4, Aik_5, Aik_6, Aik_7;
  /* For each row i of A */

  for (int i = 0; i < M; ++i){
    /* For each column j of B */ 
    for (int k = 0; k < K; k+=8) 
    {

      /* Compute C(i,j) */
      Aik = A[i+k*lda];
      Aik_1 = A[i+(k+1)*lda];
      Aik_2 = A[i+(k+2)*lda];
      Aik_3 = A[i+(k+3)*lda];
      Aik_4 = A[i+(k+4)*lda];;
      Aik_5 = A[i+(k+5)*lda];
      Aik_6 = A[i+(k+6)*lda];
      Aik_7 = A[i+(k+7)*lda];
      for (int j = 0; j < N; ++j){
        cij = Aik * B[k+j*lda];
        cij += Aik_1 * B[k+j*lda+1];
        cij += Aik_2 * B[k+j*lda+2];
        cij += Aik_3 * B[k+j*lda+3];
        cij += Aik_4 * B[k+j*lda+4];
        cij += Aik_5 * B[k+j*lda+5];
        cij += Aik_6 * B[k+j*lda+6];
        cij += Aik_7 * B[k+j*lda+7];
        C[i+j*lda] += cij;
      }
    }
  }
}

//copy array into the padded matrix
static void pad(double* restrict padA, double* restrict A, const int lda, const int newlda)
{
  for (int j = 0; j < lda; j++) {
    for (int i = 0; i < lda; i++) {
      padA[i + j*newlda] = A[i + j*lda];
    }   
  }   
}

//copy array into the unpadded matrix
static void unpad(double* restrict padA, double* restrict A, const int lda, const int newlda)
{
  for (int j = 0; j < lda; j++) {
    for (int i = 0; i < lda; i++) {
      A[i + j*lda] = padA[i + j*newlda];
    }   
  }   
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (const int lda, double* restrict A, double* restrict B, double* restrict C)
{
  int newlda = lda;
  int div = 8;
  if (lda % div){
    int t = lda % div;
    newlda = lda + (div-t);  
  }
  
  double* padA = (double*) _mm_malloc(newlda * newlda * sizeof(double), 32);
  pad(padA, A, lda, newlda);

  double* padB = (double*) _mm_malloc(newlda * newlda * sizeof(double), 32);
  pad(padB, B, lda, newlda);

  double* padC = (double*) _mm_malloc(newlda * newlda * sizeof(double), 32);
  pad(padC, C, lda, newlda);

  /* For each block-row of A */ 
  for (int i = 0; i < newlda; i += BLOCK_SIZE)
  {
    /* For each block-column of B */
    for (int j = 0; j < newlda; j += BLOCK_SIZE)
    {
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < newlda; k += BLOCK_SIZE)
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int M = min (BLOCK_SIZE, newlda-i);
        int N = min (BLOCK_SIZE, newlda-j);
        int K = min (BLOCK_SIZE, newlda-k);
        /* Perform individual block dgemm */
        do_block_unroll8(newlda, M, N, K, padA + i + k*newlda, padB + k + j*newlda, padC + i + j*newlda);

      }
    }
  }
  unpad(padC, C, lda, newlda);
}


