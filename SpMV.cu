/* 
 * Qinglei Cao
 * Hw 10 for CS594
 * SpMV 
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define SPMV 
#define DIV 10000
//#define CORRECT
//#define TRID

#if !defined(SPMV)
    #define GEMV 
#endif

__global__ void kernel_gemv(double *A_d, double *x_d, double *y_d, int m)
{
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < m) {
        double tmp = 0.0;
        for(int j = 0; j < m; j++)
            tmp += A_d[i*m+j] * x_d[j];

        y_d[i] = tmp;
    }
}

__global__ void kernel_spmv(double *value_d, double *x_d, double *y_d, int *rowptr_d, int *colidx_d, int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < m) {
        double tmp = 0.0;
        for(int j = rowptr_d[i]; j < rowptr_d[i+1]; j++){
            tmp += value_d[colidx_d[j]] * x_d[colidx_d[j]];
	}
        y_d[i] = tmp;
    }
}

void rowCount_CPU(double *A, int *row, int m){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < m; j++){
            if(0 != A[i*m+j])
                row[i] += 1;
        }
    }
}

void csr_CPU(double *A, int *colidx, double *value, int m){
    int idx = 0;
    int count = 0;
    for(int i = 0; i < m; i++){
        for(int j = 0; j < m; j++){
            if(0 != A[m*i+j]){
                colidx[idx++] = j;
                value[count++] = A[m*i+j];
	    }
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

void print_vector_d(double *d, int m){
    for(int i = 0; i < m; i++)
        printf("%lf ", d[i]);
    printf("\n");
}

void print_vector_i(int *d, int m){
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
            	A[m*i+j] = 1.0;
        }
    }
}

void set_matrix_trid(double *A, int m){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < m; j++){
            if((i==j)|| (i==j+1) || (j==i+1))
                A[m*i+j] = 1.0;
        }
    }
}

void set_vector(double *A, int m){
    for(int i = 0; i < m; i++){
        A[i] = 1.0;
    }
}

int main(int argc, char **argv){
 
    double *A;
    int *colidx, *rowptr, *row;

    if(2 != argc){
	fprintf(stderr, "Usage: ./exe matrix_size\n");
	return 1;
    }

    int m = atoi(argv[1]);
    printf("matrix size: %d ", m);

    A = (double *)calloc(m*m, sizeof(double)); 
    rowptr = (int *)calloc(m+1, sizeof(int));
    row = (int *)calloc(m, sizeof(int));

#if defined(TRID)
    set_matrix_trid(A, m);
#else
    set_matrix(A, m);
#endif

    rowCount_CPU(A, row, m);

    /* total number of nonzero */
    int sum = 0;
    for(int i = 0; i < m; i++)
	sum += row[i];
    
    colidx = (int *)calloc(sum, sizeof(int));
    double *value = (double *)calloc(sum, sizeof(double));

    /* CSR */
    for(int i = 0; i <= m; i++)
	rowptr[i+1] = rowptr[i] + row[i];
    rowptr[0] = 0;
    
    csr_CPU(A, colidx, value, m);

/********************** PART 2 ***********************************/
/*********************** SpMV ************************************/

    double *A_d, *x_d, *y_d, *value_d;
    int *colidx_d, *rowptr_d;
    double time = 0.0;
    
    double *x = (double *)calloc(m, sizeof(double));
    double *y = (double *)calloc(m, sizeof(double));

    set_vector(x, m);

    cudaMalloc((void **)&x_d, m*sizeof(double));
    cudaMalloc((void **)&y_d, m*sizeof(double));
    cudaMemcpy(x_d, x, m*sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&A_d, m*m*sizeof(double));
    cudaMemcpy(A_d, A, m*m*sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&colidx_d, sum*sizeof(int));
    cudaMemcpy(colidx_d, colidx, sum*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&rowptr_d, (m+1)*sizeof(int));
    cudaMemcpy(rowptr_d, rowptr, (m+1)*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&value_d, sum*sizeof(double));
    cudaMemcpy(value_d, value, sum*sizeof(double), cudaMemcpyHostToDevice);

    dim3 block(1024, 1);
    dim3 grid((int)ceil((double)m/(double)block.x), 1);

    /* time start */
    time = MPI_Wtime();

#if defined(GEMV)
    kernel_gemv<<<grid, block>>>(A_d, x_d, y_d, m);
#endif

#if defined(SPMV)
    kernel_spmv<<<grid, block>>>(value_d, x_d, y_d, rowptr_d, colidx_d, m);
#endif

    /* time end */
    time = MPI_Wtime() - time;
    printf("time is: %.10lf\n", time);

    cudaMemcpy(y, y_d, m*sizeof(double), cudaMemcpyDeviceToHost);

#if defined(CORRECT)
    double *y_v_d;
    double *y_v = (double *)calloc(m, sizeof(double));
    cudaMalloc((void **)&y_v_d, m*sizeof(double));
    kernel_gemv<<<grid, block>>>(A_d, x_d, y_v_d, m);
    cudaMemcpy(y_v, y_v_d, m*sizeof(double), cudaMemcpyDeviceToHost);

    for(int i = 0; i < m; i++){
        if(y[i] != y_v[i]){
            fprintf(stderr, "result is not correct\n");
            return 1;
        }
    }
    printf("Result is correct!\n\n");
#endif

/*
    printf("matrix is:\n");
    print_matrix(A, m);
    printf("x is:\n");
    print_vector_d(x, m);
    printf("result y is:\n");
    print_vector_d(y, m);
    printf("row count is:\n");
    print_vector_i(row, m);
*/

/*
#if defined GPU
    cudaFree(A_d);
    cudaFree(colidx_d);
    cudaFree(rowptr_d);
    cudaFree(row_d);
    cudaFree(x_d);
    cudaFree(y_d);
#endif

    free(A);
    free(colidx);
    free(rowptr);
    free(row);
    free(x);
    free(y);
*/

    return 0;
}
