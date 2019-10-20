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
    
    int COL = atoi(argv[2]);
	int ROW = atoi(argv[3]);
    double u_in[ROW*COL];
    
    ifstream inputFile;

	cout << "DUPA WELCOME" << " | " << argv[0] << " | " << argv[1] << " | " << argv[2] << " | " << argv[3] << endl;
	cout << "ROW: " << ROW << " | " << "COL: " << COL <<endl;
    
    inputFile.open(argv[1]);
    
    if (inputFile)
	{
		int i,j = 0;
		for (i = 0; i < ROW; i++)
		{
			for (j = 0; j < COL; j++)
			{	
				inputFile >> u_in[i*ROW+j];
			}
		}
	} else {
		cout << "Error opening the file.\n";
	}
	inputFile.close();

    
    int multi = atoi(argv[4]);
	int NX = COL*multi;
	int NY = ROW*multi;

// --- Przeliczenie hz --- //

	double sampling = 10.0 * pow(10.0, (-6)); 	// Sampling = 10 micro
	double lam = 633.0 * (pow(10.0,(-9))); 		// Lambda = 633 nm
	double k = 2.0 * M_PI / lam;				// Wektor falowy k
	double z_in = 500.0*(pow(10.0,(-3)));		// Odleglosc propagacji = 0,5 m
	double z_out = 1000.0*(pow(10.0,(-3)));     // Koniec odległości propagacji = 1 m
	double z_delta = 50.0*(pow(10.0,(-3)));     // Skok odległości = 0,05 m
	//double z = z_in+(ip*z_delta);             // Odległość Z dla każdego wątku MPI
    double z = z_in;

    printf("k = %.1f | lam = %.1f | z = %.4f mm | ", k, lam*(pow(10.0,(9))), z);
	printf(" ");




	return 0;
} 

