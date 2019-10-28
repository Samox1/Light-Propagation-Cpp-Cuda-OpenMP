#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <complex>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <omp.h>
//#include <mpi.h>

#define DATA_OFFSET_OFFSET 0x000A
#define WIDTH_OFFSET 0x0012
#define HEIGHT_OFFSET 0x0016
#define BITS_PER_PIXEL_OFFSET 0x001C
#define HEADER_SIZE 14
#define INFO_HEADER_SIZE 40
#define NO_COMPRESION 0
#define MAX_NUMBER_OF_COLORS 0
#define ALL_COLORS_REQUIRED 0


using namespace std;

const std::complex<double> i1(0, 1);
typedef unsigned int int32;
typedef short int16;
typedef unsigned char byte;


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


// ----------------------------------------------------------------------------------------------------------------------------------------------- //
// --- Functions --- Functions --- Functions --- Functions --- Functions --- Functions --- Functions --- Functions --- Functions --- Functions --- //
// ----------------------------------------------------------------------------------------------------------------------------------------------- //

void u_in_in_big(double* u_in, cufftDoubleComplex* data, int NX, int NY, int multi)
{
	for(int ii=0; ii < NY ; ii++)
	{
		for(int jj=0; jj < NX ; jj++)
		{
			data[ii*NX+jj].x = 0;
			data[ii*NX+jj].y = 0;
		}
	}

	for(int ii=0; ii < (int)NY/multi ; ii++)
	{
		for(int jj=0; jj < (int)NX/multi ; jj++)
		{
			data[(ii*NX+jj)+(NX*NY*(multi-1)/(multi*2)+NX*(multi-1)/(multi*2))].x = u_in[ii*(NX/multi)+jj];
		}
	}
}


void hz(double lam, double z, double k, double sampling, int NX, int NY, cufftDoubleComplex* hz_cutab)
{
	std::complex<double>* hz_tab;
	hz_tab = (std::complex<double> *) malloc ( sizeof(std::complex<double>)* NX * NY);

	double fi = 0;
	double teta = 0;
	double lam_z = 0;

	fi = k * z;
	teta = k / (2.0 * z);
	lam_z = lam * z;
	double quad = 0.0;
	double teta1 = 0.0;


	for(int iy=0; iy < NY; iy++)
	{
		//printf("\n");
		for(int ix=0; ix < NX ; ix++)
		{
			quad = pow(((double)ix-((double)NX/2.0))*sampling, 2) + pow(((double)iy-((double)NY/2.0))*sampling, 2);
			teta1 = teta * quad;
			//hz_tab[iy*NX+ix] = std::exp(i*fi) * std::exp(i*teta1)/(i*lam_z);
			hz_tab[iy*NX+ix] = std::exp(i1*fi) * std::exp(i1*teta1)/(i1*lam_z);
			hz_cutab[iy*NX+ix].x = hz_tab[iy*NX+ix].real();
			hz_cutab[iy*NX+ix].y = hz_tab[iy*NX+ix].imag();
			//printf("%.2f\t", hz_cutab[iy*NX+ix].x);
		}
	}
	free(hz_tab);
}


void Qroll(cufftDoubleComplex* u_in_fft, cufftDoubleComplex* data, int NX, int NY)
{
	for(int iy=0; iy<(NY/4); iy++)	//Petla na przepisanie tablicy koncowej
	{
		for(int jx=0; jx<(NX/4); jx++)
		{
			u_in_fft[(NX/2*NY/4+NY/4)+(jx+iy*NX/2)] = data[iy*(NX)+jx];		// Q1 -> Q4
			u_in_fft[(jx+NX/4)+(iy*NX/2)] = data[(iy*(NX)+jx)+(NX*NY*3/4)];		// Q3 -> Q2
			u_in_fft[(jx)+(iy*NX/2)] = data[((iy*NX)+jx)+(NX*3/4+NX*NY*3/4)];	// Q4 -> Q1
			u_in_fft[(jx)+(iy*NX/2)+NX*NY/2/4] = data[((iy*NX)+jx)+(NX*3/4)];	// Q2 -> Q3
		}
	}
}


void amplitude_print(cufftDoubleComplex* u_in_fft, int NX, int NY, FILE* fp)
{
	// --- Przeliczanie Amplitudy --- //

	for(int ii=0; ii<(NX*NY/4); ii++)
	{
		u_in_fft[ii].x = sqrt(pow(u_in_fft[ii].x, 2) + pow(u_in_fft[ii].y, 2));
	}

	double mini_data = u_in_fft[0].x;

	for(int ii=0; ii<(NX*NY/4); ii++)
	{
		if (u_in_fft[ii].x < mini_data){ mini_data = u_in_fft[ii].x; }
	}

	double max_data = u_in_fft[0].x;
	mini_data = -mini_data;

	for(int ii=0; ii<(NX*NY/4); ii++)
	{
		u_in_fft[ii].x = u_in_fft[ii].x + mini_data;
		if (u_in_fft[ii].x > max_data) { max_data = u_in_fft[ii].x; }
	}

	for(int ii=0; ii<(NX*NY/4); ii++)
	{
		if (ii%(NX/2) == 0){fprintf (fp,"\n");}
		u_in_fft[ii].x = u_in_fft[ii].x / max_data * 255.0;
		fprintf (fp,"%.0f\t", u_in_fft[ii].x);
	}
}


int FFT_Z2Z(cufftDoubleComplex* dData, int NX, int NY)
{
	// Create a 2D FFT plan.
	int err = 0;
	cufftHandle plan1;
	if (cufftPlan2d(&plan1, NX, NY, CUFFT_Z2Z) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT Error: Unable to create plan\n");
		err = -1;
	}

	if (cufftExecZ2Z(plan1, dData, dData, CUFFT_FORWARD) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
		err = -1;
	}

	if (cudaDeviceSynchronize() != cudaSuccess){
  		fprintf(stderr, "Cuda error: Failed to synchronize\n");
   		err = -1;
	}

	cufftDestroy(plan1);
	return err;
}


int IFFT_Z2Z(cufftDoubleComplex* dData, int NX, int NY)
{
	// Create a 2D FFT plan.
	int err = 0;
	cufftHandle plan1;
	if (cufftPlan2d(&plan1, NX, NY, CUFFT_Z2Z) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT Error: Unable to create plan\n");
		err = -1;
	}

	if (cufftExecZ2Z(plan1, dData, dData, CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
		err = -1;
	}

	if (cudaDeviceSynchronize() != cudaSuccess){
  		fprintf(stderr, "Cuda error: Failed to synchronize\n");
   		err = -1;
	}

	cufftDestroy(plan1);
	return err;
}


void BMP_Save_Amplitude(cufftDoubleComplex* u_out, int NX, int NY, FILE* fp)
{
  // --- SAVE BMP FILE --- //
  //uint8_t colorIndex = 0;
  //uint16_t color = 0;
  unsigned int headers[13];
  int extrabytes;
  int paddedsize;
  int x = 0;
  int y = 0;
  int n = 0;
  int red = 0;
  int green = 0;
  int blue = 0;

  int WIDTH = NX/2;
  int HEIGHT = NY/2;

  extrabytes = 4 - ((WIDTH * 3) % 4);                 // How many bytes of padding to add to each
                                                    // horizontal line - the size of which must
                                                    // be a multiple of 4 bytes.
  if (extrabytes == 4)
    extrabytes = 0;

  paddedsize = ((WIDTH * 3) + extrabytes) * HEIGHT;

// Headers...
// Note that the "BM" identifier in bytes 0 and 1 is NOT included in these "headers".

  headers[0]  = paddedsize + 54;      // bfSize (whole file size)
  headers[1]  = 0;                    // bfReserved (both)
  headers[2]  = 54;                   // bfOffbits
  headers[3]  = 40;                   // biSize
  headers[4]  = WIDTH;                // biWidth
  headers[5]  = HEIGHT;               // biHeight

// Would have biPlanes and biBitCount in position 6, but they're shorts.
// It's easier to write them out separately (see below) than pretend
// they're a single int, especially with endian issues...

  headers[7]  = 0;                    // biCompression
  headers[8]  = paddedsize;           // biSizeImage
  headers[9]  = 0;                    // biXPelsPerMeter
  headers[10] = 0;                    // biYPelsPerMeter
  headers[11] = 0;                    // biClrUsed
  headers[12] = 0;                    // biClrImportant

// outfile = fopen(filename, "wb");

  //File file = fopen("test.bmp", "wb");
  if (!fp) {
    cout << "There was an error opening the file for writing";
    //return;
  }else{

// Headers begin...
// When printing ints and shorts, we write out 1 character at a time to avoid endian issues.

	fprintf(fp, "BM");

  for (n = 0; n <= 5; n++)
  {
    fprintf(fp, "%c", headers[n] & 0x000000FF);
    fprintf(fp, "%c", (headers[n] & 0x0000FF00) >> 8);
    fprintf(fp, "%c", (headers[n] & 0x00FF0000) >> 16);
    fprintf(fp, "%c", (headers[n] & (unsigned int) 0xFF000000) >> 24);
  }

// These next 4 characters are for the biPlanes and biBitCount fields.

  fprintf(fp, "%c", 1);
  fprintf(fp, "%c", 0);
  fprintf(fp, "%c", 24);
  fprintf(fp, "%c", 0);

  for (n = 7; n <= 12; n++)
  {
    fprintf(fp, "%c", headers[n] & 0x000000FF);
    fprintf(fp, "%c", (headers[n] & 0x0000FF00) >> 8);
    fprintf(fp, "%c", (headers[n] & 0x00FF0000) >> 16);
    fprintf(fp, "%c", (headers[n] & (unsigned int) 0xFF000000) >> 24);
  }

  	// --- Przeliczanie Amplitudy --- //

	for(int ii=0; ii<(NX*NY/4); ii++)
	{
		u_out[ii].x = sqrt(pow(u_out[ii].x, 2) + pow(u_out[ii].y, 2));
	}

	double mini_data = u_out[0].x;

	for(int ii=0; ii<(NX*NY/4); ii++)
	{
		if (u_out[ii].x < mini_data){ mini_data = u_out[ii].x; }
	}

	double max_data = u_out[0].x;
	mini_data = -mini_data;

	for(int ii=0; ii<(NX*NY/4); ii++)
	{
		u_out[ii].x = u_out[ii].x + mini_data;
		if (u_out[ii].x > max_data) { max_data = u_out[ii].x; }
	}

	for(int ii=0; ii<(NX*NY/4); ii++)
	{
		//if (ii%(NX/2) == 0){fprintf (fp,"\n");}
		u_out[ii].x = u_out[ii].x / max_data * 255.0;
		//fprintf (fp,"%.0f\t", u_in_fft[ii].x);
	}


// Headers done, now write the data...

  for (y = HEIGHT - 1; y >= 0; y--)     // BMP image format is written from bottom to top...
  {
    for (x = 0; x <= WIDTH - 1; x++)
    {

		red = u_out[x+(NX/2*y)].x;
		if (red > 255) red = 255; if (red < 0) red = 0;

		green = red;
		blue = red;

      	// --- RGB range from 0 to 255 --- //
      	// if (red > 255) red = 255; if (red < 0) red = 0;
      	// if (green > 255) green = 255; if (green < 0) green = 0;
      	// if (blue > 255) blue = 255; if (blue < 0) blue = 0;

      	// Also, it's written in (b,g,r) format...
      	fprintf (fp, "%c", blue);
      	fprintf (fp, "%c", green);
      	fprintf (fp, "%c", red);
    }
    if (extrabytes)      // See above - BMP lines must be of lengths divisible by 4.
    {
      	for (n = 1; n <= extrabytes; n++)
      	{
			fprintf (fp, "%c", 0);
      	}
    }
  }

  //fclose(fp);
  cout << "Writing to BMP complete!" << endl;
  }         // --- END SAVING BMP FILE --- //
}


void ReadImage(const char *fileName,byte **pixels, int32 *width, int32 *height, int32 *bytesPerPixel)
{
        FILE *imageFile = fopen(fileName, "rb");
        int32 dataOffset;
        fseek(imageFile, DATA_OFFSET_OFFSET, SEEK_SET);
        fread(&dataOffset, 4, 1, imageFile);
        fseek(imageFile, WIDTH_OFFSET, SEEK_SET);
        fread(width, 4, 1, imageFile);
        fseek(imageFile, HEIGHT_OFFSET, SEEK_SET);
        fread(height, 4, 1, imageFile);
        int16 bitsPerPixel;
        fseek(imageFile, BITS_PER_PIXEL_OFFSET, SEEK_SET);
        fread(&bitsPerPixel, 2, 1, imageFile);
        *bytesPerPixel = ((int32)bitsPerPixel) / 8;

        int paddedRowSize = (int)(4 * ceil((float)(*width) / 4.0f))*(*bytesPerPixel);
        int unpaddedRowSize = (*width)*(*bytesPerPixel);
		int totalSize = unpaddedRowSize*(*height);
		cout << "BMP FILE: " << fileName << " | Width: " << *width << " | Height: " << *height << " | Total Size: " << totalSize << " | BitsPerPixel: " << bitsPerPixel << endl;

		*pixels = (byte*)malloc(totalSize);
        int i = 0;
        byte *currentRowPointer = *pixels+((*height-1)*unpaddedRowSize);
        for (i = 0; i < *height; i++)
        {
                fseek(imageFile, dataOffset+(i*paddedRowSize), SEEK_SET);
            fread(currentRowPointer, 1, unpaddedRowSize, imageFile);
            currentRowPointer -= unpaddedRowSize;
        }


        fclose(imageFile);
}


/*
 * compile: /usr/local/cuda/bin/nvcc -ccbin g++ -I../../common/inc -m64 -Xcompiler -fopenmp -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o cudaOpenMP -c cudaOpenMP.cu -lgomp -lcufft
 * start program: ./cudaOpenMP Test_NTO_1024.bmp 2 500.0 633.0 10.0
 * start program: ./cudaOpenMP plik_z_przezroczem.BMP  Multiply_tmp  Odleglosc_Z_mm  Dl_fali_Lambda_nm  Sampling_micro
 */


// --- Main Part --- MAIN --- MAIN --- MAIN --- MAIN --- MAIN --- MAIN --- MAIN --- MAIN --- MAIN --- MAIN --- MAIN --- MAIN --- MAIN --- MAIN --- MAIN --- //

int main(int argc, char *argv[])
{

    cout << "Light Propagation using Convolution Propagation Method and GPU" << endl;

	cout << "WELCOME: " << argv[0] << " | " << argv[1] << " | " << argv[2] << " | " << argv[3] << " | " << atoi(argv[4]) << " | " << atoi(argv[5]) << endl;

	printf("\n---------------------------\n");
	// --- PC Specs finder --- //

	int num_gpus = 0;   					// number of CUDA GPUs
	cudaGetDeviceCount(&num_gpus);
	if (num_gpus < 1)
    {
        printf("no CUDA capable devices were detected\n");
        return 1;
    }
	printf("Number of host CPUs:\t%d\n", omp_get_num_procs());
    printf("Number of CUDA devices:\t%d\n", num_gpus);

    for (int i = 0; i < num_gpus; i++)
    {
        cudaDeviceProp dprop;
        cudaGetDeviceProperties(&dprop, i);
        printf("   %d: %s\n", i, dprop.name);
    }

    printf("---------------------------\n\n");

	// --- Import BMP image --- //
	byte *pixels;
    int32 width;
    int32 height;
    int32 bytesPerPixel;
	ReadImage(argv[1], &pixels, &width, &height,&bytesPerPixel);

	double* Image_Red = (double *) malloc ( sizeof(double)* width * height);
	double* Image_Green = (double *) malloc ( sizeof(double)* width * height);
	double* Image_Blue = (double *) malloc ( sizeof(double)* width * height);

	int iterator = 0;
	for(int i=0; i<(height*width)*3; i+=3)
	{
		Image_Red[iterator]	= pixels[i];
		Image_Green[iterator] = pixels[i+1];
		Image_Blue[iterator] = pixels[i+2];
		iterator++;
	}

	free(pixels);

	int32 COL = width;
	int32 ROW = height;

	int multi = atoi(argv[2]);
	int NX = COL*multi;
	int NY = ROW*multi;

	// --- Przeliczenie hz --- //

	double sampling = atof(argv[5]) * pow(10.0, (-6)); 	// Sampling = 10 micro
	double lam = atof(argv[4]) * (pow(10.0,(-9))); 		// Lambda = 633 nm
	double k = 2.0 * M_PI / lam;						// Wektor falowy k
	double z_in = atof(argv[3])*(pow(10.0,(-3)));		// Odleglosc propagacji = 0,5 m
	double z_out = 1000.0*(pow(10.0,(-3)));     		// Koniec odległości propagacji = 1 m
	double z_delta = 50.0*(pow(10.0,(-3)));     		// Skok odległości = 0,05 m
	//double z = z_in+(ip*z_delta);             		// Odległość Z dla każdego wątku MPI
    double z = z_in;

    printf("\nVariables | k = %.1f | Lambda = %.1f nm | Z = %.4f m | Sampling = %.3f micro | Tablica tymczasowa = x%i |\n\n", k, lam*(pow(10.0,(9))), z, sampling*pow(10.0,(6)), multi);


	// --- FFT tablicy wejsciowej --- //
	cufftDoubleComplex* data;
	data = (cufftDoubleComplex *) malloc ( sizeof(cufftDoubleComplex)* NX * NY);

	cufftDoubleComplex* dData;
	cudaMalloc((void **) &dData, sizeof(cufftDoubleComplex)* NX * NY);

	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate: Allocate Cuda Memory\n");
		return -1;
	}

	size_t pitch1;

	// --- Wpisanie tablicy wejsciowej do wiekszej tablicy tymczasowej --- //
	u_in_in_big(Image_Green, data, NX, NY, multi);			// Poki co 'Image_Green' jako tablica wejsciowa

	// --- Liczenie U_in = FFT{u_in} --- //
 	cudaMallocPitch(&dData, &pitch1, sizeof(cufftDoubleComplex)*NX, NY);
	cudaMemcpy2D(dData,pitch1,data,sizeof(cufftDoubleComplex)*NX,sizeof(cufftDoubleComplex)*NX,NX,cudaMemcpyHostToDevice);

	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate: Calculate FFT{u_in}\n");
		return -1;
	}

	if (FFT_Z2Z(dData, NX, NY) == -1) {
		return -1;
	}
	cudaMemcpy(data, dData, sizeof(cufftDoubleComplex)*NX*NY, cudaMemcpyDeviceToHost);


	// --- Liczenie hz --- //
	cufftDoubleComplex* hz_tab;
	hz_tab = (cufftDoubleComplex *) malloc ( sizeof(cufftDoubleComplex)* NX * NY);
	hz(lam, z, k, sampling, NX, NY, hz_tab);


	// --- Liczenie hz = FFT{hz_tab} --- //
	cufftDoubleComplex* hz;
	cudaMalloc((void **) &hz, sizeof(cufftDoubleComplex)* NX * NY);

	size_t pitch2;
 	cudaMallocPitch(&hz, &pitch2, sizeof(cufftDoubleComplex)*NX, NY);
	cudaMemcpy2D(hz,pitch2,hz_tab,sizeof(cufftDoubleComplex)*NX,sizeof(cufftDoubleComplex)*NX,NX,cudaMemcpyHostToDevice);

	if(cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate: FFT{hz_tab}\n");
		return -1;
	}

	if (FFT_Z2Z(hz, NX, NY) == -1) {
		return -1;
	}

	// --- Do the actual multiplication --- //
	multiplyElementwise<<<NX*NY, 1>>>(dData, hz, NX*NY);


	// --- Liczenie u_out = iFFT{dData = U_OUT} --- //
	if(IFFT_Z2Z(dData, NX, NY) == -1) { return -1; }
	cudaMemcpy(data, dData, sizeof(cufftDoubleComplex)*NX*NY, cudaMemcpyDeviceToHost);

	// --- ROLL cwiartek, zeby wszystko sie zgadzalo na koniec --- //
	cufftDoubleComplex* u_out;
	u_out = (cufftDoubleComplex *) malloc (sizeof(cufftDoubleComplex)* NX/2 * NY/2);

	Qroll(u_out, data, NX, NY);

	
	// --- Zapis do pliku BMP --- //
	char filename[128];
	snprintf ( filename, 128, "z_%.3lf-m_lam_%.1lf-nm_samp_%.1lf-micro.BMP", z, lam*(pow(10.0,(9))), sampling*(pow(10.0,(6))));
	FILE* fp = fopen(filename,"wb");

	// --- Przeliczanie Amplitudy i Zapis do pliku --- //
	//amplitude_print(u_out, NX, NY, fp);
	BMP_Save_Amplitude(u_out, NX, NY, fp);

	fclose(fp);
	
	// --- Zwalnianie pamieci --- //
	cudaFree(u_out);
	cudaFree(data);
	cudaFree(dData);
	cudaFree(hz_tab);
	cudaFree(hz);

	free(Image_Red);
	free(Image_Green);
	free(Image_Blue);

	return 0;
}
