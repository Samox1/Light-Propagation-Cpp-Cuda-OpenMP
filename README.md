# Light Propagation using Convolution Propagation Method and GPU    
## Project uses C++ and CUDA (and OpenMP in future)
Author: Szymon Baczyński (Warsaw University of Technology) <br>
Date: 24.10.2019 <br><br>


Projekt opierający się o komunikację MPI i bibliotekę CUDA, by wykorzystać pełny potencjał obliczeniowy klastra DWARF wydziału Fizyki PW. <br />

Program w skrócie wykonuje się w następujących krokach:
1.  Stworzenie dwuwymiarowej tablicy o wymiarach zadanych przez użytkownika (tablica ta jest przezroczem, przez które będzie propagować się światło).
2.  Wpisanie tablicy utworzonej w punkcie 1 do dwa razy większej tablicy (mniejsza tablica umieszczona w środku większej).
3.  Nowo utworzona tablica (od teraz nazywana tablicą wejściową) kopiowana jest na kartę graficzną (GPU).
4.  Na GPU liczona jest szybka transformata Fouriera (FFT) tablicy wejściowej.
5.  Wynik wysyłany jest do procesów MPI, który następnie kopiowany jest na GPU.
6.  Każdy proces oblicza dwuwymiarową tablicę odpowiedzi impulsowej (h(z) - zależna od odległości propagacji z) i wysyła ją na GPU.
7.  GPU obliczają FFT odpowiedzi impulsowej, a następnie mnożą transformatę tablicy wejściowej z transformatą odpowiedzi impulsowej.
8.  Wynik mnożenia zostaje poddany odwrotnej transformacie Fouriera. Po wykonaniu odwrotnej transformaty, jej wynik kopiowany jest z GPU na hosta.
9.  Otrzymana tablica danych zostaje poddana operacji ROLL - zamiana miejscami ćwiartek, dla prawidłowego wyniku. 
10. Ostateczny wynik zostaje zapisany do pliku w postaci amplitudy.

### TO DO LIST:
- [x] ...
- [x] ...
- [x] ...


## Kompilacja i uruchamianie programu: 
Code for calculations of Light Propagation is in: [cudaOpenMP.cu](https://github.com/Samox1/Light-Propagation-Cpp-Cuda-OpenMP/blob/master/src/cudaOpenMP.cu) <br>
Makefile for compilation: [Makefile](https://github.com/Samox1/Light-Propagation-Cpp-Cuda-OpenMP/blob/master/src/Makefile)<br>

### Compile Command: <br>
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
