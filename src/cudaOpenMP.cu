#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <complex>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
//#include <omp.h>
//#include <mpi.h>

using namespace std;

__global__ void multiplyElementwise(cufftDoubleComplex* f0, cufftDoubleComplex* f1, int size)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < size)
    {
        double a, b, c, d;
        a = f0[i].x; 
        b = f0[i].y;
        c = f1[i].x; 
        d = f1[i].y;
        f0[i].x = a*c - b*d;
        f0[i].y = a*d + b*c;
    }
}


// --- Main Part --- //

int main(int argc, char *argv[])
{

	cout << "Welcome to CUDA test" << endl;
	
	return 0;
} 

