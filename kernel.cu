#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Initialization.h"
#include "Initialization_GPU.cu"
#include "PrintToFile.h"
#include "MatrixOperation.cu"

void SelectGPU()
{
	int i;
	cudaDeviceProp prop;
	cudaGetDeviceCount(&i);
	if(i==1)
		printf("    There is %d GPU device on your PC.\n",i);
	else
		printf("    There are %d GPU devices on your PC.\n",i);
	cudaGetDeviceProperties(&prop,0);
	printf("    Device %d is: %s.  Compute capability: %d.%d, SMs = %d\n",0,prop.name,prop.major,prop.minor,prop.multiProcessorCount);
	printf("    maxThreadsPerBlock = %d, maxThreadsDim = [%d,%d,%d], maxGridSize = [%d,%d,%d]\n",prop.maxThreadsPerBlock,prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2],prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
	cudaSetDevice(0);
	printf("    Device %d is chosen.\n\n",0);
}

double SumDouble(double *Array,int n)
{
	double Sum=0.0;
	for(int i=0;i<n;i++)
	{
		Sum += Array[i];
	}
	return Sum;
}

int  main()
{
	int GridDim = 80,BlockDim = 256;
	SelectGPU();
	printf("This is my first CUDA code.\n");

	//************ �ļ���д �������� ***********//
	char *Directory1,*Directory2;
	Directory1 = "Array.txt";
	Directory2 = "Matrix.txt";

	//************ ��ʱ�� �������� ***********//
	int START_CLOCK,END_CLOCK;
	double Iter_Running_Time,Total_Running_Time;

	//************ CG �������� ***********//
	int IterationNum;
	int n = 20;
	int Bandwidth = 5;

	double *a,*b,*c,*PartialSum;
	double *dev_a,*dev_b,*dev_c,*dev_PartialSum;

	a = (double*)malloc(n*sizeof(double));
	b = (double*)malloc(n*sizeof(double));
	c = (double*)malloc(n*sizeof(double));
	PartialSum = (double*)malloc(GridDim*sizeof(double));
	cudaMalloc((void**)&dev_a,n*sizeof(double));
	cudaMalloc((void**)&dev_b,n*sizeof(double));
	cudaMalloc((void**)&dev_c,n*sizeof(double));
	cudaMalloc((void**)&dev_PartialSum,GridDim*sizeof(double));

	InitializeArray(a,n,1.0);
	InitializeArray(b,n,0.0);
	MatrixMultiply_Banded(a,b,n,2*Bandwidth-1,Bandwidth);
	PrintArray(a,Directory1,"a",n);
	PrintArray(b,Directory1,"b",n);

	InitializeArray_GPU<<<GridDim,BlockDim>>>(dev_a,n,1.0);
	InitializeArray_GPU<<<GridDim,BlockDim>>>(dev_b,n,0.0);
	MatrixMultiply_GPU<<<GridDim,BlockDim>>>(dev_a,dev_b,n,2*Bandwidth-1,Bandwidth);
	cudaMemcpy(a,dev_a,n*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(b,dev_b,n*sizeof(double),cudaMemcpyDeviceToHost);
	//cudaMemcpy(dev_a,a,n*sizeof(double),cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_b,dev_a,n*sizeof(double),cudaMemcpyDeviceToDevice);
	PrintArray(a,Directory1,"a",n);
	PrintArray(b,Directory1,"b",n);

	InitializeArray_GPU<<<GridDim,BlockDim>>>(dev_a,n,1.0);
	InitializeArray_GPU<<<GridDim,BlockDim>>>(dev_b,n,1.0);
	InitializeArray_GPU<<<GridDim,BlockDim>>>(dev_c,n,0.0);
	Dotproduct<<<GridDim,BlockDim>>>(dev_a,dev_b,dev_c,n);
	cudaMemcpy(c,dev_c,n*sizeof(double),cudaMemcpyDeviceToHost);
	PrintArray(c,Directory1,"c",n);
	printf("Dot Product = %lf\n",SumDouble(c,n));

	InitializeArray_GPU<<<GridDim,1>>>(dev_PartialSum,GridDim,0.0);
	Dotproduct_Shared_Reduction<<<GridDim,BlockDim>>>(dev_a,dev_b,dev_PartialSum,n);
	cudaMemcpy(PartialSum,dev_PartialSum,GridDim*sizeof(double),cudaMemcpyDeviceToHost);
	printf("Dot Product = %lf\n",SumDouble(PartialSum,GridDim));

	InitializeArray_GPU<<<GridDim,1>>>(dev_PartialSum,GridDim,0.0);
	START_CLOCK = clock();
	for(int i=0;i<1000;i++)
	{
		Pi_Shared_Reduction<<<GridDim,BlockDim>>>(dev_PartialSum,1000000000);
	}
	cudaMemcpy(PartialSum,dev_PartialSum,GridDim*sizeof(double),cudaMemcpyDeviceToHost);
	END_CLOCK = clock();
	Total_Running_Time = END_CLOCK - START_CLOCK;
	printf("Pi = %22.18lf\n",4.0*SumDouble(PartialSum,GridDim));
	printf("Time = %16.12lf s\n",(double)(END_CLOCK - START_CLOCK)/CLOCKS_PER_SEC);

	return 0;
}
