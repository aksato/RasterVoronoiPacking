#include "gpuinfo.h"

#include <iostream> // to output to the console
#include <cuda.h> // to get memory on the device
#include <cuda_runtime.h> // to get device count

#define BLOCK_SIZE 16
#define REDUCTION_BLOCK_SIZE 256
#define EPS 0.000001

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		int a;
		std::cin >> a;
		if (abort) exit(code);
	}
}

namespace CUDAPACKING {

	// Problem pointers
	CudaRasterNoFitPolygon **h_dpointerdpointers, **h_hpointerdpointers, **d_dpointerdpointer;
	int *d_itemTypeMap;
	float *d_overlapmap;
	// Solution pointers
	int *d_posx, *d_posy, *d_angles;
	float *d_weights;

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

	void allocItemTypes(int numItems) {
		gpuErrchk(cudaMalloc((void**)&d_itemTypeMap, numItems*sizeof(int)));
	}

	void alloDevicecSolutionPointers(int numItems) {
		gpuErrchk(cudaMalloc((void**)&d_posx, numItems*sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&d_posy, numItems*sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&d_angles, numItems*sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&d_weights, numItems*sizeof(float)));
	}

	void setItemType(int itemId, int typeId) {
		gpuErrchk(cudaMemcpy(d_itemTypeMap + itemId, &typeId, sizeof(int), cudaMemcpyHostToDevice));
	}

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

	void allocDeviceMaxIfp(size_t memSize) {
		gpuErrchk(cudaMalloc((void**)&d_overlapmap, memSize));
	}

	// GPU displaced sum of two matrix. TODO: Store nfp widths, heights and origins in shared memory.
	__global__ static void DisplacedSumKernel(float *d_overlapmap, int omwidth, int omheight, int overlapmapx, int overlapmapy, int nAngles, CudaRasterNoFitPolygon **nfpSet, int nfpcount, int *itemType, int itemId, int itemAngle, int *posx, int *posy, int *angles)
	{
		const int tidi = blockDim.x * blockIdx.x + threadIdx.x;
		const int tidj = blockDim.y * blockIdx.y + threadIdx.y;
		const int orbitingId = itemType[itemId]*nAngles + itemAngle;

		int nfpCoordx, nfpCoordy;
		float tempVal = 0;
		if (tidi < omwidth && tidj < omheight) {
			for (int k = 0; k < nfpcount; k++) {
				if (k == itemId) continue;
				int staticId = itemType[k] * nAngles + angles[k];
				nfpCoordx = tidi - overlapmapx - posx[k] + nfpSet[staticId][orbitingId].origin.x;
				nfpCoordy = tidj - overlapmapy - posy[k] + nfpSet[staticId][orbitingId].origin.y;
				if (nfpCoordx >= 0 && nfpCoordx < nfpSet[staticId][orbitingId].m_width && nfpCoordy >= 0 && nfpCoordy < nfpSet[staticId][orbitingId].m_height)
					tempVal += (float)nfpSet[staticId][orbitingId].matrix[nfpCoordy*nfpSet[staticId][orbitingId].m_width + nfpCoordx];
			}
			d_overlapmap[tidj*omwidth + tidi] = tempVal;
		}
	}

	// GPU displaced sum of two matrix with weights. TODO: Store nfp widths, heights and origins in shared memory.
	__global__ static void DisplacedWeightedSumKernel(float *d_overlapmap, int omwidth, int omheight, int overlapmapx, int overlapmapy, int nAngles, CudaRasterNoFitPolygon **nfpSet, int nfpcount, int *itemType, int itemId, int itemAngle, int *posx, int *posy, int *angles, float *weights)
	{
		const int tidi = blockDim.x * blockIdx.x + threadIdx.x;
		const int tidj = blockDim.y * blockIdx.y + threadIdx.y;
		const int orbitingId = itemType[itemId] * nAngles + itemAngle;

		int nfpCoordx, nfpCoordy;
		float tempVal = 0;
		if (tidi < omwidth && tidj < omheight) {
			for (int k = 0; k < nfpcount; k++) {
				if (k == itemId) continue;
				int staticId = itemType[k] * nAngles + angles[k];
				nfpCoordx = tidi - overlapmapx - posx[k] + nfpSet[staticId][orbitingId].origin.x;
				nfpCoordy = tidj - overlapmapy - posy[k] + nfpSet[staticId][orbitingId].origin.y;
				if (nfpCoordx >= 0 && nfpCoordx < nfpSet[staticId][orbitingId].m_width && nfpCoordy >= 0 && nfpCoordy < nfpSet[staticId][orbitingId].m_height)
					tempVal += weights[k]*(float)nfpSet[staticId][orbitingId].matrix[nfpCoordy*nfpSet[staticId][orbitingId].m_width + nfpCoordx];
			}
			d_overlapmap[tidj*omwidth + tidi] = tempVal;
		}
	}

	__global__ void FindMinimumPositionKernel(float *g_idata, float *g_odata, int *g_opos, int n, bool first) {
		extern __shared__ float smem[];

		float *sdata = smem;
		int *sdatapos = (int*)&sdata[REDUCTION_BLOCK_SIZE];

		// each thread loads one element from global to shared mem
		unsigned int tid = threadIdx.x;
		unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
		if (i < n) {
			sdata[tid] = g_idata[i];
			if (first) sdatapos[tid] = i; else sdatapos[tid] = g_opos[i];
		}
		else {
			sdata[tid] = g_idata[n - 1];
			if (first) sdatapos[tid] = n - 1; else sdatapos[tid] = g_opos[n - 1];
		}
		__syncthreads();

		// do reduction in shared mem
		for (unsigned int s = blockDim.x / 2; s>0; s >>= 1) {
			if (tid < s) {
				if (abs(sdata[tid] - sdata[tid + s]) < FLT_EPSILON) {
					if(sdatapos[tid + s] < sdatapos[tid]) {
						sdata[tid] = sdata[tid + s];
						sdatapos[tid] = sdatapos[tid + s];
					}
				}
				else {
					if (sdata[tid + s] < sdata[tid]) {
						sdata[tid] = sdata[tid + s];
						sdatapos[tid] = sdatapos[tid + s];
					}
				}
			}
			__syncthreads();
		}

		// write result for this block to global mem
		if (tid == 0) { g_odata[blockIdx.x] = sdata[0]; g_opos[blockIdx.x] = sdatapos[0]; }
	}

	// Copy position and angles to device
	void uploadSolutionParameters(int *posx, int *posy, int *angles, int nItems) {
		cudaMemcpy(d_posx, posx, nItems*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_posy, posy, nItems*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_angles, angles, nItems*sizeof(int), cudaMemcpyHostToDevice);
	}

	// Copy position, angles and weights to device
	void uploadSolutionParameters(int *posx, int *posy, int *angles, float *weights, int nItems) {
		cudaMemcpy(d_posx, posx, nItems*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_posy, posy, nItems*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_angles, angles, nItems*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_weights, weights, nItems*sizeof(float), cudaMemcpyHostToDevice);
	}

	// Create overlap map on device
	void detOverlapMapOnDevice(int curItem, int curItemAngle, int nItems, int numAngles, int overlapmap_width, int overlapmap_height, int overlapmapx, int overlapmapy, bool useGlsWeights) {
		// Execute Kernel to determine the map
		dim3 blocks(1, 1, 1);
		dim3 threadsperblock(BLOCK_SIZE, BLOCK_SIZE, 1);
		blocks.x = ((overlapmap_width / BLOCK_SIZE) + (((overlapmap_width) % BLOCK_SIZE) == 0 ? 0 : 1));
		blocks.y = ((overlapmap_height / BLOCK_SIZE) + (((overlapmap_height) % BLOCK_SIZE) == 0 ? 0 : 1));
		if (!useGlsWeights) DisplacedSumKernel << <blocks, threadsperblock >> >(d_overlapmap, overlapmap_width, overlapmap_height, overlapmapx, overlapmapy, numAngles, d_dpointerdpointer, nItems, d_itemTypeMap, curItem, curItemAngle, d_posx, d_posy, d_angles);
		else DisplacedWeightedSumKernel << <blocks, threadsperblock >> >(d_overlapmap, overlapmap_width, overlapmap_height, overlapmapx, overlapmapy, numAngles, d_dpointerdpointer, nItems, d_itemTypeMap, curItem, curItemAngle, d_posx, d_posy, d_angles, d_weights);
	}

	// Returns a pointer to an overlap map on host
	float *getcuOverlapMap(int curItem, int curItemAngle, int nItems, int numAngles, int overlapmap_width, int overlapmap_height, int overlapmapx, int overlapmapy, int *posx, int *posy, int *angles, float *weights, bool useGlsWeights) {
		// Upload solution parameters to GPU
		if (useGlsWeights) uploadSolutionParameters(posx, posy, angles, weights, nItems);
		else uploadSolutionParameters(posx, posy, angles, nItems);

		// Determine overlap map on GPU
		detOverlapMapOnDevice(curItem, curItemAngle, nItems, numAngles, overlapmap_width, overlapmap_height, overlapmapx, overlapmapy, useGlsWeights);

		// Copy overlap map result to host
		float *h_overlapmap = (float *)malloc(overlapmap_width*overlapmap_height*sizeof(float));
		cudaMemcpy(h_overlapmap, d_overlapmap, overlapmap_width * overlapmap_height * sizeof(float), cudaMemcpyDeviceToHost);
		return h_overlapmap;
	}

	// Create overlap map on device
	void detOverlapMapOnDevice(float **d_valueVec, int **d_PosVec, float *d_map, int width, int height) {
		// Execute Kernel to determine the minimum position
		dim3 blocks2(1, 1, 1);
		dim3 threadsperblock2(REDUCTION_BLOCK_SIZE, 1, 1);
		blocks2.x = ((width*height) / REDUCTION_BLOCK_SIZE) + (((width*height) % REDUCTION_BLOCK_SIZE) == 0 ? 0 : 1);
		gpuErrchk(cudaMalloc((void**) &(*d_valueVec), blocks2.x*sizeof(float)));
		gpuErrchk(cudaMalloc((void**) &(*d_PosVec), blocks2.x*sizeof(int)));
		FindMinimumPositionKernel << <blocks2, threadsperblock2, REDUCTION_BLOCK_SIZE*sizeof(float)+REDUCTION_BLOCK_SIZE*sizeof(int) >> >(d_map, *d_valueVec, *d_PosVec, width*height, true);
		int reducedsize = blocks2.x;
		while (reducedsize > 1) {
			blocks2.x = ((reducedsize / REDUCTION_BLOCK_SIZE) + (((reducedsize) % REDUCTION_BLOCK_SIZE) == 0 ? 0 : 1));
			FindMinimumPositionKernel << <blocks2, threadsperblock2, REDUCTION_BLOCK_SIZE*sizeof(float)+REDUCTION_BLOCK_SIZE*sizeof(int) >> >(*d_valueVec, *d_valueVec, *d_PosVec, reducedsize, false);
			reducedsize = blocks2.x;
		}
	}

	// Returns minimum overlap position and value using GPU. TODO: Change placement in gpu memory.
	float getcuMinimumOverlap(int curItem, int curItemAngle, int nItems, int numAngles, int overlapmap_width, int overlapmap_height, int overlapmapx, int overlapmapy, int *posx, int *posy, int *angles, float *weights, int &minX, int &minY, bool useGlsWeights) {
		// Upload solution parameters to GPU
		if (useGlsWeights) uploadSolutionParameters(posx, posy, angles, weights, nItems);
		else uploadSolutionParameters(posx, posy, angles, nItems);

		// Determine overlap map on GPU
		detOverlapMapOnDevice(curItem, curItemAngle, nItems, numAngles, overlapmap_width, overlapmap_height, overlapmapx, overlapmapy, useGlsWeights);

		// Execute Kernel to determine the minimum position
		float *d_temp_output; int *d_temp_pos;
		detOverlapMapOnDevice(&d_temp_output, &d_temp_pos, d_overlapmap, overlapmap_width, overlapmap_height);

		// Copy result to host
		float minVal; int linearPosition; 
		cudaMemcpy(&minVal, d_temp_output, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&linearPosition, d_temp_pos, sizeof(int), cudaMemcpyDeviceToHost);
		minX = - overlapmapx + linearPosition % overlapmap_width; minY = - overlapmapy + linearPosition / overlapmap_width;

		// Free GPU temporary pointers
		gpuErrchk(cudaFree(d_temp_output));
		gpuErrchk(cudaFree(d_temp_pos));

		return minVal;
	}
}