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
	int *d_itemTypeMap, *h_itemTypeMap;
	float *d_overlapmap;
	size_t d_overlapmapSize;
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
		h_itemTypeMap = (int*)malloc(numItems*sizeof(int));
	}

	void alloDevicecSolutionPointers(int numItems) {
		gpuErrchk(cudaMalloc((void**)&d_posx, numItems*sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&d_posy, numItems*sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&d_angles, numItems*sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&d_weights, numItems*sizeof(float)));
	}

	void setItemType(int itemId, int typeId) {
		gpuErrchk(cudaMemcpy(d_itemTypeMap + itemId, &typeId, sizeof(int), cudaMemcpyHostToDevice));
		h_itemTypeMap[itemId] = typeId;
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
		d_overlapmapSize = memSize;
	}

	bool reallocDeviceMaxIfp(size_t memSize) {
		if (memSize > d_overlapmapSize) {
			gpuErrchk(cudaFree(d_overlapmap));
			gpuErrchk(cudaMalloc((void**)&d_overlapmap, memSize));
			d_overlapmapSize = memSize;
			return true;
		}
		return false;
	}

	// GPU displaced sum of two matrix with weights.
	__global__ static void DisplacedSingleWeightedSumKernel(float *d_overlapmap, int omwidth, int omheight, int overlapmapx, int overlapmapy,
														CudaRasterNoFitPolygon **nfpSet, int staticId, int orbitingId, 
														int posx, int posy, float weight,
														int rectBLx, int rectBLy, int rectTRx, int rectTRy) {
		const int tidi = rectBLx + blockDim.x * blockIdx.x + threadIdx.x;
		const int tidj = rectBLy + blockDim.y * blockIdx.y + threadIdx.y;

		if (tidi <= rectTRx && tidj <= rectTRy) {
			int nfpCoordx, nfpCoordy;
			nfpCoordx = tidi - overlapmapx - posx + nfpSet[staticId][orbitingId].origin.x;
			nfpCoordy = tidj - overlapmapy - posy + nfpSet[staticId][orbitingId].origin.y;
			d_overlapmap[tidj*omwidth + tidi] += weight*(float)nfpSet[staticId][orbitingId].matrix[nfpCoordy*nfpSet[staticId][orbitingId].m_width + nfpCoordx];
		}
	}

	// GPU displaced sum of the layout. TODO: Store nfp widths, heights and origins in shared memory.
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

	// GPU displaced sum of the layout with weights.
	__global__ static void DisplacedWeightedSumKernel(float *d_overlapmap, int omwidth, int omheight, int overlapmapx, int overlapmapy, int nAngles, CudaRasterNoFitPolygon **nfpSet, int nfpcount, int *itemType, int itemId, int itemAngle, int *posx, int *posy, int *angles, float *weights)
	{
		extern __shared__ int smem0[];
		int *sposx = smem0;
		int *sposy = &sposx[nfpcount];
		int *soriginx = &sposy[nfpcount];;
		int *soriginy = &soriginx[nfpcount];;
		int *swidth = &soriginy[nfpcount];;
		int *sheight = &swidth[nfpcount];;
		float *sweights = (float*)&sheight[nfpcount];
		int *sstaticid = (int*)&sweights[nfpcount];

		//const int tid = threadIdx.x + blockDim.x*threadIdx.y + (blockIdx.x*blockDim.x*blockDim.y) + (blockIdx.y*blockDim.x*blockDim.y);
		const int tid = threadIdx.y * blockDim.x + threadIdx.x;
		const int orbitingId = itemType[itemId] * nAngles + itemAngle;
		
		if      (tid <     nfpcount) sposx[tid] = posx[tid];
		else if (tid < 2 * nfpcount) sposy[tid - nfpcount] = posy[tid - nfpcount];
		else if (tid < 3 * nfpcount) { int k = tid - 2 * nfpcount; int staticId = itemType[k] * nAngles + angles[k]; soriginx[k] = nfpSet[staticId][orbitingId].origin.x;}
		else if (tid < 4 * nfpcount) { int k = tid - 3 * nfpcount; int staticId = itemType[k] * nAngles + angles[k]; soriginy[k] = nfpSet[staticId][orbitingId].origin.y; }
		else if (tid < 5 * nfpcount) { int k = tid - 4 * nfpcount; int staticId = itemType[k] * nAngles + angles[k]; swidth[k] = nfpSet[staticId][orbitingId].m_width; }
		else if (tid < 6 * nfpcount) { int k = tid - 5 * nfpcount; int staticId = itemType[k] * nAngles + angles[k]; sheight[k] = nfpSet[staticId][orbitingId].m_height; }
		else if (tid < 7 * nfpcount) sweights[tid - 6 * nfpcount] = weights[tid - 6 * nfpcount];
		else if (tid < 8 * nfpcount) { int k = tid - 7 * nfpcount; sstaticid[tid - 7 * nfpcount] = itemType[k] * nAngles + angles[k]; }
		__syncthreads();
		//if (tid == 0) 
		//	for (int i = 0; i < nfpcount; i++)
		//		printf("%d: %d %d %d %d %d %d %f %d\n", i, sposx[i], sposy[i], soriginx[i], soriginy[i], swidth[i], sheight[i], sweights[i], sstaticid[i]);

		const int tidi = blockDim.x * blockIdx.x + threadIdx.x;
		const int tidj = blockDim.y * blockIdx.y + threadIdx.y;
		int nfpCoordx, nfpCoordy;
		float tempVal = 0;
		if (tidi < omwidth && tidj < omheight) {
			for (int k = 0; k < nfpcount; k++) {
				if (k == itemId) continue;
				//int staticId = itemType[k] * nAngles + angles[k];
				//nfpCoordx = tidi - overlapmapx - posx[k] + nfpSet[staticId][orbitingId].origin.x;
				//nfpCoordy = tidj - overlapmapy - posy[k] + nfpSet[staticId][orbitingId].origin.y;
				nfpCoordx = tidi - overlapmapx - sposx[k] + soriginx[k];
				nfpCoordy = tidj - overlapmapy - sposy[k] + soriginy[k];
				//if (nfpCoordx >= 0 && nfpCoordx < nfpSet[staticId][orbitingId].m_width && nfpCoordy >= 0 && nfpCoordy < nfpSet[staticId][orbitingId].m_height)
				if (nfpCoordx >= 0 && nfpCoordx < swidth[k] && nfpCoordy >= 0 && nfpCoordy < sheight[k])
					//tempVal += weights[k]*(float)nfpSet[staticId][orbitingId].matrix[nfpCoordy*nfpSet[staticId][orbitingId].m_width + nfpCoordx];
					tempVal += sweights[k] * (float)nfpSet[sstaticid[k]][orbitingId].matrix[nfpCoordy*swidth[k] + nfpCoordx];
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
		else DisplacedWeightedSumKernel << <blocks, threadsperblock, 7*nItems*sizeof(int)+nItems*sizeof(float)>> >(d_overlapmap, overlapmap_width, overlapmap_height, overlapmapx, overlapmapy, numAngles, d_dpointerdpointer, nItems, d_itemTypeMap, curItem, curItemAngle, d_posx, d_posy, d_angles, d_weights);
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

	bool getLimits(int relativeOriginx, int relativeOriginy, int vmWidth, int vmHeight, int overlapmap_width, int overlapmap_height, int &rectBLx, int &rectBLy, int &rectTRx, int &rectTRy) {
		rectBLx = relativeOriginx < 0 ? 0 : relativeOriginx;
		rectTRy = relativeOriginy + vmHeight > overlapmap_height ? overlapmap_height - 1 : relativeOriginy + vmHeight - 1;
		rectTRx = relativeOriginx + vmWidth > overlapmap_width ? overlapmap_width - 1 : relativeOriginx + vmWidth - 1;
		rectBLy = relativeOriginy < 0 ? 0 : relativeOriginy;

		if (rectTRx< 0 || rectTRy < 0 ||
			rectBLx > overlapmap_width || rectBLy > overlapmap_height) return false;
		return true;
	}

	__global__ void initKernel(float * devPtr, const float val, const size_t nwords)
	{
		int tidx = threadIdx.x + blockDim.x * blockIdx.x;
		int stride = blockDim.x * gridDim.x;

		for (; tidx < nwords; tidx += stride)
			devPtr[tidx] = val;
	}

	// Returns a pointer to an overlap map on host using an alternative method
	float *getcuOverlapMap2(int curItem, int curItemAngle, int nItems, int numAngles, int overlapmap_width, int overlapmap_height, int overlapmapx, int overlapmapy, int *posx, int *posy, int *angles, float *weights, bool useGlsWeights) {
		// Upload solution parameters to GPU
		if (useGlsWeights) uploadSolutionParameters(posx, posy, angles, weights, nItems);
		else uploadSolutionParameters(posx, posy, angles, nItems);
		
		// Initialize overlap map with zero values
		dim3 blocks(1, 1, 1);
		dim3 threadsperblock(BLOCK_SIZE, BLOCK_SIZE, 1);
		blocks.x = ((overlapmap_width / BLOCK_SIZE) + (((overlapmap_width) % BLOCK_SIZE) == 0 ? 0 : 1));
		blocks.y = ((overlapmap_height / BLOCK_SIZE) + (((overlapmap_height) % BLOCK_SIZE) == 0 ? 0 : 1));
		initKernel << <blocks, threadsperblock >> >(d_overlapmap, 10.0, overlapmap_width*overlapmap_height);

		// Determine overlap map on GPU
		int itemId, staticId, orbitingId;
		orbitingId = h_itemTypeMap[curItem] * numAngles + angles[curItem];
		for (itemId = 0; itemId < nItems; itemId++) {
			if (itemId == curItem) continue;
			staticId = h_itemTypeMap[itemId] * numAngles + angles[itemId];
			int relativeOriginx = overlapmapx + posx[itemId] - h_hpointerdpointers[staticId][orbitingId].origin.x;
			int relativeOriginy = overlapmapy + posy[itemId] - h_hpointerdpointers[staticId][orbitingId].origin.y;
			
			// Determine the rectangular area of influence
			int rectBLx, rectBLy, rectTRx, rectTRy;
			if (!getLimits(relativeOriginx, relativeOriginy, h_hpointerdpointers[staticId][orbitingId].m_width, h_hpointerdpointers[staticId][orbitingId].m_height, overlapmap_width, overlapmap_height, rectBLx, rectBLy, rectTRx, rectTRy)) return NULL;
			int rectW = rectTRx - rectBLx + 1; int rectH = rectTRy - rectBLy + 1;

			// Determine block size
			blocks.x = ((rectW / BLOCK_SIZE) + (((rectW) % BLOCK_SIZE) == 0 ? 0 : 1));
			blocks.y = ((rectH / BLOCK_SIZE) + (((rectH) % BLOCK_SIZE) == 0 ? 0 : 1));

			// Execute Kernel to determine the map
			if (!useGlsWeights) weights[itemId] = 1.0;
			DisplacedSingleWeightedSumKernel << <blocks, threadsperblock >> >(d_overlapmap, overlapmap_width, overlapmap_height, overlapmapx, overlapmapy,
				d_dpointerdpointer, staticId, orbitingId,
				posx[itemId], posy[itemId], weights[itemId],
				rectBLx, rectBLy, rectTRx, rectTRy);
		}
		// Copy overlap map result to host
		float *h_overlapmap = (float *)malloc(overlapmap_width*overlapmap_height*sizeof(float));
		cudaMemcpy(h_overlapmap, d_overlapmap, overlapmap_width * overlapmap_height * sizeof(float), cudaMemcpyDeviceToHost);
		return h_overlapmap;
	}


	// Create overlap map on device
	void findMinimumOnDevice(float **d_valueVec, int **d_PosVec, float *d_map, int width, int height) {
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
		findMinimumOnDevice(&d_temp_output, &d_temp_pos, d_overlapmap, overlapmap_width, overlapmap_height);

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