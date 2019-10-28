# Light Propagation using Convolution Propagation Method and GPU    
## Project uses C++ and CUDA (and OpenMP in future)
Author: Szymon Baczyński (Warsaw University of Technology) <br>
Date: 24.10.2019 <br><br>

The program is briefly carried out in the following steps ([paper](https://github.com/Samox1/Propagation-C-CUDA-NTO-2019/blob/master/B_01_199504_OptComm.PDF)):
1. Creation of a two-dimensional array with BMP-defined dimensions (the array is a slide through which light will propagate).
2. Enter the table created in step 1 to X (usually twice) as large a table (smaller table placed in the center of a larger one).
3. The newly created table (now called the input table) is copied to the graphics card (GPU).
4. The Fast Fourier Transform (FFT) of the input table is counted on the GPU.
5. CPU calculates a two-dimensional impulse response table (h(z) - depending on the propagation distance z) and sends it to the GPU.
6. GPU calculate the FFT of the impulse response and then multiply the input table transform with the impulse response transform.
7. The result of multiplication is subjected to the Inverse Fourier Transform. After performing the IFFT, its result is copied from the GPU to the host.
8. The resulting data table is subjected to the ROLL operation - quadrant swapping for the correct result.
9. The final result is saved to a file (BMP - Grayscale) in the form of an amplitude.


### TO DO LIST:
- [x] Import TXT file with "HOLE" as 0-255 Grayscale
- [x] Perform calculation on CPU and GPU (CUDA)
- [x] Export TXT file after calculations
- [x] Import BMP (take only Green channel)
- [x] Export BMP (as Grayscale)
- [x] Check CUDA compatible devices
- [ ] Clean Code
- [ ] Slice STL files


## Kompilacja i uruchamianie programu: 
Code for calculations of Light Propagation is in: [cudaOpenMP.cu](https://github.com/Samox1/Light-Propagation-Cpp-Cuda-OpenMP/blob/master/src/cudaOpenMP.cu) <br>
Makefile for compilation (from CUDA example): [Makefile](https://github.com/Samox1/Light-Propagation-Cpp-Cuda-OpenMP/blob/master/src/Makefile)<br>

### Simple Compile Command: <br>
/usr/local/cuda/bin/nvcc -ccbin g++ -I../../common/inc  -m64    -Xcompiler -fopenmp -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o cudaOpenMP1 -c cudaOpenMP.cu -lgomp -lcufft
<br>

### Start Command: <br>
##### Template command: [XXX]$ ./cudaOpenMP BMP_in Multi Z_in Lambda Sampling <br>
##### Example command:  [XXX]$ ./cudaOpenMP Test_NTO_1024.bmp 2 500.0 633.0 10.0 <br>

BMP_in - BMP file as "HOLE" <br>
Multi - Multiplier for temporary array (usually =2) (int) <br>
Z_in - Propagation Distance in milimeters [mm] (double) <br>
Lambda - Wavelength in nanometers [nm] (double) <br>
Sampling - Space between each pixels in micrometers [microm] (double) <br><br>

### Output File: <br>
Template for Output file: z_%.3lf-m_lam_%.1lf-nm_samp_%.1lf-micro.BMP  <br>
Example output file:      z_0.500-m_lam_450.0-nm_samp_10.0-micro.BMP  <br>
