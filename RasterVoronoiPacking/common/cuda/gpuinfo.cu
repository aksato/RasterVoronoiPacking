#include "gpuinfo.h"

#include <iostream> // to output to the console
#include <cuda.h> // to get memory on the device
#include <cuda_runtime.h> // to get device count

#define BLOCK_SIZE 16

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

namespace CUDAPACKING {

	bool getTotalMemory(int &gpuDeviceCount, size_t &free, size_t &total) {
		int deviceCount, device;
		gpuDeviceCount = 0;
		struct cudaDeviceProp properties;
		cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
		if (cudaResultCode != cudaSuccess)
			deviceCount = 0;
		/* machines with no GPUs can still report one emulation device */
		for (device = 0; device < deviceCount; ++device) {
			cudaGetDeviceProperties(&properties, device);
			if (properties.major != 9999) /* 9999 means emulation only */
				++gpuDeviceCount;
		}
		//printf("%d GPU CUDA device(s) found\n", gpuDeviceCount);

		/* don't just return the number of gpus, because other runtime cuda
		errors can also yield non-zero return values */
		if (gpuDeviceCount > 0) {
			cudaMemGetInfo(&free, &total);
			//std::cout << "free memory: " << free / 1024 / 1024 << "mb, total memory: " << total / 1024 / 1024 << "mb" << std::endl;
			return true; /* success */
		}
		else return false; /* failure */
	}

	CudaRasterNoFitPolygon **h_dpointerdpointers, **h_hpointerdpointers, **d_dpointerdpointer;

	void allocHostNfpPointers(int numItems, int numOrientations) {
		// Nofit polygon values
		h_dpointerdpointers = (CudaRasterNoFitPolygon**)malloc(numItems*numOrientations*sizeof(CudaRasterNoFitPolygon*));
		h_hpointerdpointers = (CudaRasterNoFitPolygon**)malloc(numItems*numOrientations*sizeof(CudaRasterNoFitPolygon*));
		for (int l = 0; l < numItems*numOrientations; l++)
			h_hpointerdpointers[l] = (CudaRasterNoFitPolygon*)malloc(numItems*numOrientations*sizeof(CudaRasterNoFitPolygon));
	}

	void allocSingleDeviceNfpMatrix(int staticId, int orbitingId, int *matrix, int width, int height, int originx, int originy) {
		h_hpointerdpointers[staticId][orbitingId].setOrigin(cuPoint(originx, originy));
		h_hpointerdpointers[staticId][orbitingId].setWidth(width);
		h_hpointerdpointers[staticId][orbitingId].setHeight(height);
		gpuErrchk(cudaMalloc((void**)&h_hpointerdpointers[staticId][orbitingId].matrix, width*height*sizeof(int)));
		gpuErrchk(cudaMemcpy(h_hpointerdpointers[staticId][orbitingId].matrix, matrix, width*height*sizeof(int), cudaMemcpyHostToDevice));
	}

	void allocDeviceNfpPointers(int numItems, int numOrientations) {
		for (int l = 0; l < numItems*numOrientations; l++) {
			gpuErrchk(cudaMalloc((void**)&h_dpointerdpointers[l], numItems*numOrientations*sizeof(CudaRasterNoFitPolygon)));
			gpuErrchk(cudaMemcpy(h_dpointerdpointers[l], h_hpointerdpointers[l], numItems*numOrientations*sizeof(CudaRasterNoFitPolygon), cudaMemcpyHostToDevice));
		}
		gpuErrchk(cudaMalloc((void**)&d_dpointerdpointer, numItems*numOrientations*sizeof(CudaRasterNoFitPolygon*)));
		gpuErrchk(cudaMemcpy(d_dpointerdpointer, h_dpointerdpointers, numItems*numOrientations*sizeof(CudaRasterNoFitPolygon*), cudaMemcpyHostToDevice));
	}

	__global__ static void DisplacedSumKernel(float *d_overlapmap, int omwidth, int omheight, int overlapmapx, int overlapmapy, int nAngles, CudaRasterNoFitPolygon **nfpSet, int nfpcount, int itemId, int itemAngle, int *posx, int *posy, int *angles)
	{
		const int tidi = blockDim.x * blockIdx.x + threadIdx.x;
		const int tidj = blockDim.y * blockIdx.y + threadIdx.y;
		const int orbitingId = itemId*nAngles + itemAngle;

		int nfpCoordx, nfpCoordy;
		float tempVal = 0;
		if (tidi < omwidth && tidj < omheight) {
			for (int k = 0; k < nfpcount; k++) {
				if (k == itemId) continue;
				int staticId = k*nAngles + angles[k];
				nfpCoordx = tidi - overlapmapx - posx[k] + nfpSet[staticId][orbitingId].origin.x;
				nfpCoordy = tidj - overlapmapy - posy[k] + nfpSet[staticId][orbitingId].origin.y;
				if (nfpCoordx >= 0 && nfpCoordx < nfpSet[staticId][orbitingId].m_width && nfpCoordy >= 0 && nfpCoordy < nfpSet[staticId][orbitingId].m_height)
					tempVal += (float)nfpSet[staticId][orbitingId].matrix[nfpCoordy*nfpSet[staticId][orbitingId].m_width + nfpCoordx];
			}
			d_overlapmap[tidj*omwidth + tidi] = tempVal;
		}
	}

	float *getcuOverlapMap(int curItem, int curItemAngle, int nItems, int numAngles, int overlapmap_width, int overlapmap_height, int overlapmapx, int overlapmapy, int *posx, int *posy, int *angles) {
		// Create timer
		//cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);

		// Create overlap map on device
		float *d_overlapmap;
		cudaMalloc((void**)&d_overlapmap, overlapmap_width*overlapmap_height*sizeof(float));

		// Copy position and angles to device
		int *d_posx, *d_posy, *d_angles;
		cudaMalloc((void**)&d_posx, nItems*sizeof(int));
		cudaMalloc((void**)&d_posy, nItems*sizeof(int));
		cudaMalloc((void**)&d_angles, nItems*sizeof(int));
		cudaMemcpy(d_posx, posx, nItems*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_posy, posy, nItems*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_angles, angles, nItems*sizeof(int), cudaMemcpyHostToDevice);

		// Execute Kernel
		dim3 blocks(1, 1, 1);
		dim3 threadsperblock(BLOCK_SIZE, BLOCK_SIZE, 1);
		blocks.x = ((overlapmap_width / BLOCK_SIZE) + (((overlapmap_width) % BLOCK_SIZE) == 0 ? 0 : 1));
		blocks.y = ((overlapmap_height / BLOCK_SIZE) + (((overlapmap_height) % BLOCK_SIZE) == 0 ? 0 : 1));
		//cudaEventRecord(start);
		DisplacedSumKernel << <blocks, threadsperblock >> >(d_overlapmap, overlapmap_width, overlapmap_height, overlapmapx, overlapmapy, numAngles, d_dpointerdpointer, nItems, curItem, curItemAngle, d_posx, d_posy, d_angles);
		//cudaEventRecord(stop);

		// Stop timer and print execution time
		//cudaThreadSynchronize();
		//cudaEventSynchronize(stop);
		//float fmilliseconds = 0; cudaEventElapsedTime(&fmilliseconds, start, stop);
		//std::cout << "Elapsed Time: " << fmilliseconds << " ms" << std::endl;

		// Copy overlap map result to host
		float *h_overlapmap = (float *)malloc(overlapmap_height*overlapmap_height*sizeof(float));
		cudaMemcpy(h_overlapmap, d_overlapmap, overlapmap_width * overlapmap_height * sizeof(float), cudaMemcpyDeviceToHost);

		return h_overlapmap;
	}
}