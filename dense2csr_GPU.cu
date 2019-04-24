/* 
 * Qinglei Cao
 * Hw 10 for CS594
 * Dense to CSR
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define DIV 1000
#define GPU
#define CORRECT

#if !defined(GPU)
    #define CPU
#endif

void rowCount_CPU(double *A, int *row, int m){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < m; j++){
            if(0 != A[i*m+j])
                row[i] += 1;
        }
    }
}

__global__ void rowCount_GPU(double *A_d, int *row_d, int m){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < m){
    int tmp = 0.0;
    	for(int j = 0; j < m; j++){
            if(A_d[m*i+j])
	        tmp += 1;
        }
	row_d[i] = tmp;
    }
}

void csr_CPU(double *A, int *colidx, int m){
    int idx = 0;
    for(int i = 0; i < m; i++){
        for(int j = 0; j < m; j++){
            if(A[m*i+j])
                colidx[idx++] = j;
        }
    }
}

__global__ void csr_GPU(double *A_d, int *rowptr_d, int *colidx_d, int m){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < m){
    	for(int j = 0; j < m; j++){
            if(A_d[m*i+j])
	       colidx_d[rowptr_d[i]++] = j;
        }
    }
}

void print_matrix(double *A, int m){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < m; j++){
            printf(" %lf",A[i*m+j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_vector(int *d, int m){
    for(int i = 0; i < m; i++)
        printf("%d ", d[i]);
    printf("\n");
}

void set_matrix(double *A, int m){
    int r;

    srand(time(NULL));
    for(int i = 0; i < m; i++){
        for(int j = 0; j < m; j++){
	    r = (int)rand(); 
	    if(0 == (r%DIV))
            	A[m*i+j] = 2.0;
        }
    }
}

int main(int argc, char **argv){
 
    double *A;
    int *colidx, *rowptr, *row;
    double time;

#if defined(GPU)
    double *A_d;
    int *colidx_d, *rowptr_d, *row_d;
#endif

    if(2 != argc){
	fprintf(stderr, "Usage: ./exe matrix_size\n");
	return 1;
    }

    int m = atoi(argv[1]);
    printf("matrix size: %d ", m);

    A = (double *)calloc(m*m, sizeof(double)); 
    rowptr = (int *)calloc(m+1, sizeof(int));
    row = (int *)calloc(m, sizeof(int));

    set_matrix(A, m);
    //printf("The matrix is:\n");
    //print_matrix(A, m);

    /* time start */
    time = MPI_Wtime();

    /* 1) identify nonzero of each row */
#if defined(GPU)
    dim3 block(1024, 1);
    dim3 grid((int)ceil((double)m/(double)block.x), 1);
    cudaMalloc((void **)&A_d, m*m*sizeof(double));
    cudaMalloc((void **)&row_d, m*sizeof(int));
    cudaMemcpy(A_d, A, m*m*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(row_d, row, m*sizeof(int), cudaMemcpyHostToDevice);

    rowCount_GPU<<<grid, block>>>(A_d, row_d, m);

    cudaMemcpy(row, row_d, m*sizeof(int), cudaMemcpyDeviceToHost);
#endif

#if defined(CPU)
    rowCount_CPU(A, row, m);
#endif

    //print_vector(row, m);

    /* total number of nonzero */
    int sum = 0;
    for(int i = 0; i < m; i++)
	sum += row[i];
    
    //printf("sum = %d\n", sum);
    colidx = (int *)calloc(sum, sizeof(int));

    /* CSR */
    for(int i = 0; i <= m; i++)
	rowptr[i+1] = rowptr[i] + row[i];
    rowptr[0] = 0;
    
    //print_vector(rowptr, m+1);

#if defined(GPU)
    cudaMalloc((void **)&rowptr_d, (m+1)*sizeof(int));
    cudaMalloc((void **)&colidx_d, sum*sizeof(int));
    cudaMemcpy(rowptr_d, rowptr, (m+1)*sizeof(int), cudaMemcpyHostToDevice);

    csr_GPU<<<grid, block>>>(A_d, rowptr_d, colidx_d, m);

    cudaMemcpy(colidx, colidx_d, sum*sizeof(int), cudaMemcpyDeviceToHost);
#endif

#if defined(CPU)
    csr_CPU(A, colidx, m);
#endif

    /* time end */
    time = MPI_Wtime() - time;
    printf("time is: %e\n", time);

#if defined(CORRECT)
    printf("\nVerify the result: ");
    int *row_v = (int *)calloc(m, sizeof(int));
    int *rowptr_v = (int *)calloc(m+1, sizeof(int));

    rowCount_CPU(A, row_v, m);

    int sum_v = 0;
    for(int i = 0; i < m; i++)
        sum_v += row_v[i];

    int *colidx_v = (int *)calloc(sum_v, sizeof(int));

    /* CSR */
    for(int i = 0; i <= m; i++)
        rowptr_v[i+1] = rowptr_v[i] + row_v[i];
    rowptr[0] = 0;

    csr_CPU(A, colidx_v, m);

    //printf("sum_v: %d\n", sum_v);
    //printf("sum: %d\n", sum);
    if(sum != sum_v){
	fprintf(stderr, "result is not correct, sum not equal\n");
	return 1;
    }

    for(int i = 0; i < m+1; i++){
	if(rowptr[i] != rowptr_v[i]){
	    fprintf(stderr, "result is not correct, rowptr not the same\n");
	    return 1;
	}
    }

    for(int i = 0; i < sum; i++){
        if(colidx[i] != colidx_v[i]){
            fprintf(stderr, "result is not correct, colidx not the same\n");
            return 1;
        }
    }

    printf("Result is correct!\n\n");

#endif

    //printf("rowptr is:\n");
    //print_vector(rowptr, m+1);
    //printf("colidx is:\n");
    //print_vector(colidx, sum);

/*
#if defined GPU
    cudaFree(A_d);
    cudaFree(colidx_d);
    cudaFree(rowptr_d);
    cudaFree(row_d);
#endif

    free(A);
    free(colidx);
    free(rowptr);
    free(row);
*/

    return 0;
}
