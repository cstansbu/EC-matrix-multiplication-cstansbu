/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "support.h"

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {

	//@@ Insert code to implement matrix multiplication here
	//@@ You have to use shared memory for this MP
	__shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
	__shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Identify the row and column of the C element to work on
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

	float Pvalue = 0;

	// Set the max for m below
	int m_max = (int)ceil(numAColumns/(double)TILE_WIDTH);

	// Loop over the M and N tiles required to compute the Pvalue
	for (int m = 0; m < m_max; m++) {
		// Collaborative loading of A and B tiles into shared memory
		// Make sure thread is in bounds for matrix A
		// if we are outside of the matrix, set to 0
		if (Row < numARows && m * TILE_WIDTH + tx < numAColumns)
			subTileM[ty][tx] = A[Row * numAColumns + m * TILE_WIDTH + tx];
		else
			subTileM[ty][tx] = 0;

		// Make sure thread is in bounds for matrix B
		// If we are outside of the matrix, set to 0
		if (Col < numBColumns && (m * TILE_WIDTH + ty) < numBRows)
			subTileN[ty][tx] = B[(m * TILE_WIDTH + ty) * numBColumns + Col];
		else
			subTileN[ty][tx] = 0;

	  __syncthreads();

	  for (int k = 0; k < TILE_WIDTH; ++k)
		  Pvalue += subTileM[ty][k] * subTileN[k][tx];

	  __syncthreads();
	}

	// make sure we aren't trying to write to an invalid space
	if (Row < numCRows && Col < numCColumns)
		C[Row * numCColumns + Col] = Pvalue;
}

// serial implementation
// this is basically a copy of VERIFY from support.cu tweaked to set C
void matrixMultiplySerial(float * A, float * B, float * C, int m, int k, int n) {

	for(int row = 0; row < m; ++row) {
		for(int col = 0; col < n; ++col) {
			C[row*n + col] = 0;

			for(unsigned int i = 0; i < k; ++i) {
				C[row*n + col] += A[row*k + i] * B[i*n + col];
			}
		}
	}
}

int main (int argc, char *argv[])
{

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the variables..."); fflush(stdout);

    float *A_h, *B_h, *C_h, *CSerial_h;
    float *A_d, *B_d, *C_d;
    size_t A_sz, B_sz, C_sz;
    unsigned matArow, matAcol;
    unsigned matBrow, matBcol;
    unsigned matCrow, matCcol;
    
    // loop variables
    bool keepGoing = true;
    int ITERATIONS = 3;

    // average arrays
    float *serialAvg, *kernelAvg;
    serialAvg = (float*) malloc( sizeof(float)*ITERATIONS );
    kernelAvg = (float*) malloc( sizeof(float)*ITERATIONS );


    // initial matrix size
    int MAT_SIZE = 20;

    // loop until serial operation takes 4 minutes
    while(keepGoing){

		// Set initial amounts
		matArow = matCrow = MAT_SIZE;
		matAcol = matBrow = MAT_SIZE;
		matBcol = matCcol = MAT_SIZE;

		printf("Testing matrix size: %d\n", MAT_SIZE); fflush(stdout);

		// Give it three rounds
		for(int k = 0; k < ITERATIONS; k++){
			printf("Iteration: %d\n", k); fflush(stdout);

			// Set matrix sizes
			A_sz = matArow*matAcol;
			B_sz = matBrow*matBcol;
			C_sz = matArow*matBcol;

			A_h = (float*) malloc( sizeof(float)*A_sz );
			for (unsigned int i=0; i < A_sz; i++) { A_h[i] = (rand()%100)/100.00; }

			B_h = (float*) malloc( sizeof(float)*B_sz );
			for (unsigned int i=0; i < B_sz; i++) { B_h[i] = (rand()%100)/100.00; }

			C_h = (float*) malloc( sizeof(float)*C_sz );
			CSerial_h = (float*) malloc( sizeof(float)*C_sz );

			printf("    A: %u x %u\n    B: %u x %u\n    C: %u x %u\n", matArow, matAcol,
				matBrow, matBcol, matCrow, matCcol);

			// Allocate device variables ----------------------------------------------

			// allocate memory and check for errors
			CHECK_CUDA_RESULT(cudaMalloc(&A_d, A_sz * sizeof(float)));
			CHECK_CUDA_RESULT(cudaMalloc(&B_d, B_sz * sizeof(float)));
			CHECK_CUDA_RESULT(cudaMalloc(&C_d, C_sz * sizeof(float)));

			CHECK_CUDA_RESULT(cudaDeviceSynchronize());

			// Copy host variables to device ------------------------------------------

			// copying from host to device
			CHECK_CUDA_RESULT(cudaMemcpy(A_d, A_h, A_sz * sizeof(float), cudaMemcpyHostToDevice));
			CHECK_CUDA_RESULT(cudaMemcpy(B_d, B_h, B_sz * sizeof(float), cudaMemcpyHostToDevice));
			CHECK_CUDA_RESULT(cudaMemcpy(C_d, C_h, C_sz * sizeof(float), cudaMemcpyHostToDevice));

			CHECK_CUDA_RESULT(cudaDeviceSynchronize());

			// Performing serial calculation
			printf("Testing serial..."); fflush(stdout);
			startTime(&timer);
			matrixMultiplySerial(A_h, B_h, CSerial_h, matArow, matAcol, matBrow);
			stopTime(&timer); printf("%f s\n", elapsedTime(timer));

			serialAvg[k] = elapsedTime(timer);

			// Launch kernel
			printf("Launching kernel..."); fflush(stdout);

			//@@ Initialize the grid and block dimensions here
			dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
			dim3 dimGrid((int)ceil(matCcol/(double)TILE_WIDTH), (int)ceil(matCrow/(double)TILE_WIDTH));

			//@@ Launch the GPU Kernel here
			startTime(&timer);
			matrixMultiplyShared<<<dimGrid, dimBlock>>>(A_d, B_d, C_d,
														matArow, matAcol, matBrow, matBcol,
														matCrow, matCcol);

			cuda_ret = cudaDeviceSynchronize();
			if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
			stopTime(&timer); printf("%f s\n", elapsedTime(timer));

			kernelAvg[k] = elapsedTime(timer);

			// Copy device variables from host ----------------------------------------

			// copying results from device to host
			CHECK_CUDA_RESULT(cudaMemcpy(C_h, C_d, C_sz * sizeof(float), cudaMemcpyDeviceToHost));

			cudaDeviceSynchronize();

			// Free memory ------------------------------------------------------------

			free(A_h);
			free(B_h);
			free(C_h);
			free(CSerial_h);

			cudaFree(A_d);
			cudaFree(B_d);
			cudaFree(C_d);
		} // End iterations

		float serialTot = 0;
		float kernelTot = 0;

		// compute average
		for(int i = 0; i < ITERATIONS; i++){
			serialTot += serialAvg[i];
			kernelTot += kernelAvg[i];
		}

		serialTot = serialTot / ITERATIONS;
		kernelTot = kernelTot / ITERATIONS;

		printf("Serial average: %fs\n",serialTot);
		printf("Kernel average: %fs\n", kernelTot);

		// if serial total is greater than 240, bail
		if(serialTot > 240)
			keepGoing = false;
		else
			MAT_SIZE = MAT_SIZE * 2;
    }

   // free average loops
   free(serialAvg);
   free(kernelAvg);

   return 0;
}
