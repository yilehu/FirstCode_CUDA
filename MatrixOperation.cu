void MatrixMultiply_Banded(double *x,double *b,int m,int n,int Bandwidth)
{
	int i,j;
	int j_start,j_end;
	double A;

	for(i=0;i<m;i++)
	{
  	if(i>=0 && i<Bandwidth-1)
		{j_start = Bandwidth-1-i;j_end = n;}
		else if(i>=Bandwidth-1 && i<m-Bandwidth+1)
		{j_start = 0;j_end = n;}
		else if(i>=m-Bandwidth+1 && i<m)
		{j_start = 0;j_end = Bandwidth-1+m-i;}
  	b[i] = 0.0;
  	for(j=j_start;j<j_end;j++)
  	{
  		A = Bandwidth - abs(j-Bandwidth+1);
  		b[i] += A*x[i-(Bandwidth-1)+j];
		}
	}
}

__global__ void MatrixMultiply_GPU(double *x,double *b,int m,int n,int Bandwidth)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int j,j_start,j_end;
	double A;

	while(tid<m)
	{
  	if(tid>=0 && tid<Bandwidth-1)
		{j_start = Bandwidth-1-tid;j_end = n;}
		else if(tid>=Bandwidth-1 && tid<m-Bandwidth+1)
		{j_start = 0;j_end = n;}
		else if(tid>=m-Bandwidth+1 && tid<m)
		{j_start = 0;j_end = Bandwidth-1+m-tid;}
  	b[tid] = 0.0;
  	for(j=j_start;j<j_end;j++)
  	{
  		A = Bandwidth - abs(j-Bandwidth+1);
  		b[tid] += A*x[tid-(Bandwidth-1)+j];
		}
		tid += gridDim.x*blockDim.x;
	}
}

__global__ void Dotproduct(double *a,double *b,double *c,int Dim)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while(tid<Dim)
	{
		c[tid] = a[tid]*b[tid];
		tid += gridDim.x*blockDim.x;
	}
}

__global__ void Dotproduct_Shared(double *a,double *b,double *PartialSum,int Dim)
{
	__shared__ double cache[256];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;
	double temp = 0.0;

	while(tid<Dim)
	{
		temp += a[tid] * b[tid];
		tid += gridDim.x*blockDim.x;
	}
	cache[cacheIndex] = temp;

	__syncthreads();

	if(cacheIndex == 0)
	{
		temp = 0.0;
		for(int i=0;i<256;i++)
		{
			temp += cache[i];
		}
		PartialSum[blockIdx.x] = temp;
	}
}

__global__ void Dotproduct_Shared_Reduction(double *a,double *b,double *PartialSum,int Dim)
{
	__shared__ double cache[256];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;
	double temp = 0.0;

	while(tid<Dim)
	{
		temp += a[tid] * b[tid];
		tid += gridDim.x*blockDim.x;
	}
	cache[cacheIndex] = temp;

	__syncthreads();

	int i = 256;
	while(i>1)
	{
		if(cacheIndex < i/2)
		{
			cache[cacheIndex] += cache[cacheIndex + i/2];
		}
		i /= 2;
		__syncthreads();
	}

	if(cacheIndex == 0)
	{
		PartialSum[blockIdx.x] = cache[cacheIndex];
	}
}

__global__ void Pi_Shared_Reduction(double *PartialSum,int Dim)
{
	__shared__ double cache[256];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;
	double temp = 0.0;

	while(tid<Dim)
	{
		temp += 1.0 / (4.0*tid + 1.0) - 1.0 / (4.0*tid + 3.0);
		tid += gridDim.x*blockDim.x;
	}
	cache[cacheIndex] = temp;

	__syncthreads();

	int i = 256;
	while(i>1)
	{
		if(cacheIndex < i/2)
		{
			cache[cacheIndex] += cache[cacheIndex + i/2];
		}
		i /= 2;
		__syncthreads();
	}

	if(cacheIndex == 0)
	{
		PartialSum[blockIdx.x] = cache[cacheIndex];
	}
}
