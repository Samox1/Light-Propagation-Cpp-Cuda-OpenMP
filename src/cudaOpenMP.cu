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


// void u_in_in_big(double* u_in, cufftDoubleComplex* data, int NX, int NY, int multi);
// void h_z(double lam, double z, double k, double sampling, int NX, int NY, cufftDoubleComplex* h_z_cutab);
// void Q_roll(cufftDoubleComplex* u_in_fft, cufftDoubleComplex* data, int NX, int NY);
// void amplitude_print(cufftDoubleComplex* u_in_fft, int NX, int NY, FILE* fp);
// int FFT_Z2Z(cufftDoubleComplex* dData, int NX, int NY);
// int IFFT_Z2Z(cufftDoubleComplex* dData, int NX, int NY);



// ----------------------------------------------------------------------------------------------------------------------------------------------- //
// --- Functions --- Functions --- Functions --- Functions --- Functions --- Functions --- Functions --- Functions --- Functions --- Functions --- //
// ----------------------------------------------------------------------------------------------------------------------------------------------- //

// void u_in_in_big(double* u_in, cufftDoubleComplex* data, int NX, int NY, int multi)
// {
// 	for(int ii=0; ii < NY ; ii++)
// 	{
// 		for(int jj=0; jj < NX ; jj++)
// 		{
// 			data[ii*NX+jj].x = 0;
// 			data[ii*NX+jj].y = 0;
// 		}
// 	}

// 	for(int ii=0; ii < (int)NY/multi ; ii++)
// 	{
// 		for(int jj=0; jj < (int)NX/multi ; jj++)
// 		{
// 			data[(ii*NX+jj)+(NX*NY*(multi-1)/(multi*2)+NX*(multi-1)/(multi*2))].x = u_in[ii*(NX/multi)+jj];
// 		}
// 	}
// }


// void hz(double lam, double z, double k, double sampling, int NX, int NY, cufftDoubleComplex* hz_cutab)
// {
// 	std::complex<double>* hz_tab;
// 	hz_tab = (std::complex<double> *) malloc ( sizeof(std::complex<double>)* NX * NY);

// 	double fi = k * z;
// 	double teta = k / (2.0 * z);
// 	double lam_z = lam * z;
// 	double quad = 0.0;
// 	double teta1 = 0.0;	

// 	for(int iy=0; iy < NY; iy++)
// 	{
// 		//printf("\n");
// 		for(int ix=0; ix < NX ; ix++)
// 		{
// 			quad = pow(((double)ix-((double)NX/2.0))*sampling, 2) + pow(((double)iy-((double)NY/2.0))*sampling, 2);
// 			teta1 = teta * quad;
// 			hz_tab[iy*NX+ix] = exp(1i*fi)*exp(1i*teta1)/(1i*lam_z);
// 			hz_cutab[iy*NX+ix].x = hz_tab[iy*NX+ix].real();
// 			hz_cutab[iy*NX+ix].y = hz_tab[iy*NX+ix].imag();
// 			//printf("%.2f\t", hz_cutab[iy*NX+ix].x);
// 		}
// 	}	
// 	free(hz_tab);
// }


// void Qroll(cufftDoubleComplex* u_in_fft, cufftDoubleComplex* data, int NX, int NY)
// {
// 	for(int iy=0; iy<(NY/4); iy++)	//Petla na przepisanie tablicy koncowej
// 	{
// 		for(int jx=0; jx<(NX/4); jx++)
// 		{
// 			u_in_fft[(NX/2*NY/4+NY/4)+(jx+iy*NX/2)] = data[iy*(NX)+jx];		// Q1 -> Q4
// 			u_in_fft[(jx+NX/4)+(iy*NX/2)] = data[(iy*(NX)+jx)+(NX*NY*3/4)];		// Q3 -> Q2
// 			u_in_fft[(jx)+(iy*NX/2)] = data[((iy*NX)+jx)+(NX*3/4+NX*NY*3/4)];	// Q4 -> Q1
// 			u_in_fft[(jx)+(iy*NX/2)+NX*NY/2/4] = data[((iy*NX)+jx)+(NX*3/4)];	// Q2 -> Q3
// 		}
// 	}
// }

// void amplitude_print(cufftDoubleComplex* u_in_fft, int NX, int NY, FILE* fp)
// {
// 	// --- Przeliczanie Amplitudy --- //

// 	for(int ii=0; ii<(NX*NY/4); ii++)
// 	{	
// 		u_in_fft[ii].x = sqrt(pow(u_in_fft[ii].x, 2) + pow(u_in_fft[ii].y, 2));
// 	}
	
// 	double mini_data = u_in_fft[0].x;
	
// 	for(int ii=0; ii<(NX*NY/4); ii++)
// 	{		
// 		if (u_in_fft[ii].x < mini_data){ mini_data = u_in_fft[ii].x; }
// 	}
	
// 	double max_data = u_in_fft[0].x;
// 	mini_data = -mini_data;
	
// 	for(int ii=0; ii<(NX*NY/4); ii++)
// 	{		
// 		u_in_fft[ii].x = u_in_fft[ii].x + mini_data;
// 		if (u_in_fft[ii].x > max_data) { max_data = u_in_fft[ii].x; }
// 	}

// 	for(int ii=0; ii<(NX*NY/4); ii++)
// 	{	
// 		if (ii%(NX/2) == 0){fprintf (fp,"\n");}
// 		u_in_fft[ii].x = u_in_fft[ii].x / max_data * 255.0;
// 		fprintf (fp,"%.0f\t", u_in_fft[ii].x);
// 	}
// }

// int FFT_Z2Z(cufftDoubleComplex* dData, int NX, int NY)
// {
// 	// Create a 2D FFT plan. 
// 	int err = 0;
// 	cufftHandle plan1;
// 	if (cufftPlan2d(&plan1, NX, NY, CUFFT_Z2Z) != CUFFT_SUCCESS){
// 		fprintf(stderr, "CUFFT Error: Unable to create plan\n");
// 		err = -1;	
// 	}

// 	if (cufftExecZ2Z(plan1, dData, dData, CUFFT_FORWARD) != CUFFT_SUCCESS){
// 		fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
// 		err = -1;		
// 	}

// 	if (cudaDeviceSynchronize() != cudaSuccess){
//   		fprintf(stderr, "Cuda error: Failed to synchronize\n");
//    		err = -1;
// 	}	
	
// 	cufftDestroy(plan1);
// 	return err;
// }

// int IFFT_Z2Z(cufftDoubleComplex* dData, int NX, int NY)
// {
// 	// Create a 2D FFT plan.
// 	int err = 0; 
// 	cufftHandle plan1;
// 	if (cufftPlan2d(&plan1, NX, NY, CUFFT_Z2Z) != CUFFT_SUCCESS){
// 		fprintf(stderr, "CUFFT Error: Unable to create plan\n");
// 		err = -1;	
// 	}

// 	if (cufftExecZ2Z(plan1, dData, dData, CUFFT_INVERSE) != CUFFT_SUCCESS){
// 		fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
// 		err = -1;		
// 	}

// 	if (cudaDeviceSynchronize() != cudaSuccess){
//   		fprintf(stderr, "Cuda error: Failed to synchronize\n");
//    		err = -1;
// 	}

// 	cufftDestroy(plan1);	
// 	return err;
// }


// --- Main Part --- //

int main(int argc, char *argv[])
{

    cout << "Welcome to CUDA test" << endl;

    int COL = atoi(argv[2]);
	int ROW = atoi(argv[3]);
	//int COL = 1024;
	//int ROW = 1024;
	//double u_in[ROW*COL];
	//cout << "DEBUG" << endl;
	double* u_in;
	u_in = (double *) malloc ( sizeof(double)* COL * ROW);


	//cout << "DUPA WELCOME" << " | " << argv[0] << " | " << argv[1] << " | " << endl;
	cout << "DUPA WELCOME" << " | " << argv[0] << " | " << argv[1] << " | " << argv[2] << " | " << argv[3] << " | " << atoi(argv[4]) << endl;
	//cout << "ROW: " << ROW << " | " << "COL: " << COL <<endl;


	ifstream inputFile;
    inputFile.open(argv[1]);

    if (inputFile)
	{
		cout << "Import file: " << argv[1] << endl;
		int i,j = 0;
		for (i = 0; i < ROW; i++)
		{
			for (j = 0; j < COL; j++)
			{
				inputFile >> u_in[i*ROW+j];
			}
		}
		cout << "Import file - complete" << endl;
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

    printf("k = %.1f | lam = %.1f nm | z = %.4f m | \n", k, lam*(pow(10.0,(9))), z);

// 	// --- FFT tablicy wejsciowej --- //
// 	cufftDoubleComplex* data;
// 	data = (cufftDoubleComplex *) malloc ( sizeof(cufftDoubleComplex)* NX * NY);

// 	cufftDoubleComplex* dData;
// 	cudaMalloc((void **) &dData, sizeof(cufftDoubleComplex)* NX * NY);

// 	if (cudaGetLastError() != cudaSuccess){
// 		fprintf(stderr, "Cuda error: Failed to allocate\n");
// 		return -1;
// 	}
	
// 	size_t pitch1;

// 	u_in_in_big(u_in, data, NX, NY, multi);

// 	// Liczenie U_in = FFT{u_in}
//  	cudaMallocPitch(&dData, &pitch1, sizeof(cufftDoubleComplex)*NX, NY);
// 	cudaMemcpy2D(dData,pitch1,data,sizeof(cufftDoubleComplex)*NX,sizeof(cufftDoubleComplex)*NX,NX,cudaMemcpyHostToDevice);
 	
// 	if (cudaGetLastError() != cudaSuccess){
// 		fprintf(stderr, "Cuda error: Failed to allocate\n");
// 		return -1;	
// 	}

// 	if (FFT_Z2Z(dData, NX, NY) == -1) { return -1; }
// 		cudaMemcpy(data, dData, sizeof(cufftDoubleComplex)*NX*NY, cudaMemcpyDeviceToHost);
// 	}	
	
// // Liczenie hz

// 	cufftDoubleComplex* hz_tab;
// 	hz_tab = (cufftDoubleComplex *) malloc ( sizeof(cufftDoubleComplex)* NX * NY);
// 	hz(lam, z, k, sampling, NX, NY, hz_tab);	

// // --- Liczenie hz = FFT{hz_tab} --- //
	
// 	cufftDoubleComplex* hz;
// 	cudaMalloc((void **) &hz, sizeof(cufftDoubleComplex)* NX * NY);

// 	size_t pitch2;
//  	cudaMallocPitch(&hz, &pitch2, sizeof(cufftDoubleComplex)*NX, NY);
// 	cudaMemcpy2D(hz,pitch2,hz_tab,sizeof(cufftDoubleComplex)*NX,sizeof(cufftDoubleComplex)*NX,NX,cudaMemcpyHostToDevice);

// 	if(cudaGetLastError() != cudaSuccess){
// 		fprintf(stderr, "Cuda error: Failed to allocate\n");
// 		return -1;	
// 	}

// 	if (FFT_Z2Z(hz, NX, NY) == -1) { return -1; }

// 	// Do the actual multiplication

// 	multiplyElementwise<<<NX*NY, 1>>>(dData, hz, NX*NY);
	

// // --- Liczenie u_out = iFFT{dData = U_OUT} --- //

// 	if(IFFT_Z2Z(dData, NX, NY) == -1) { return -1; }

// 	cudaMemcpy(data, dData, sizeof(cufftDoubleComplex)*NX*NY, cudaMemcpyDeviceToHost);

// 	//printf( "\nCUFFT vals: \n");
	
// // Czytanie calosci


// // --- ROLL cwiartek, zeby wszystko sie zgadzalo na koniec --- //

// 	cufftDoubleComplex* u_out;
// 	u_out = (cufftDoubleComplex *) malloc (sizeof(cufftDoubleComplex)* NX/2 * NY/2);

// 	Qroll(u_out, data, NX, NY);

// // --- Przeliczanie Amplitudy --- //

// 	char filename[128];
// 	snprintf ( filename, 128, "result_z_%.5lf.txt", z );
// 	FILE* fp = fopen(filename,"w");

// 	amplitude_print(u_out, NX, NY, fp);

// 	fclose(fp);

// 	cudaFree(u_out);
// 	cudaFree(data);
// 	cudaFree(dData);
// 	cudaFree(hz_tab);
// 	cudaFree(hz);

	free(u_in);

	return 0;
}


