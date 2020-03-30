#include "totaloverlapmapcuda.h"
using namespace RASTERVORONOIPACKING;

#define THREADS_PER_BLOCK 512

// TODO: More efficient parallel minimum algorithm
__global__ void findMinimumDevice(quint32* d, int n, quint32 maxVal, quint32* min, quint32* pos) {
	extern __shared__ quint32 s[];
	quint32* sm = s;
	quint32* im = &sm[blockDim.x];

	uint tid = threadIdx.x;
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	sm[tid] = i < n ? d[i] : maxVal;
	im[tid] = i;

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (tid < stride) {
			if (sm[tid + stride] < sm[tid] ||
				(sm[tid] == sm[tid + stride] && im[tid + stride] < im[tid])) {
				im[tid] = im[tid + stride];
				sm[tid] = sm[tid + stride];
			}
		}
		__syncthreads();
	}
	if (tid == 0) {
		min[blockIdx.x] = sm[0];
		pos[blockIdx.x] = im[0];
	}
}

// TODO: More efficient parallel minimum algorithm
__global__ void findMinimumDevice2(int n, quint32 maxVal, quint32* min, quint32* pos) {
	extern __shared__ quint32 s[];
	quint32* sm = s;
	quint32* im = &sm[blockDim.x];

	uint tid = threadIdx.x;
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	sm[tid] = i < n ? min[i] : maxVal;
	im[tid] = i < n ? pos[i] : -1;

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (tid < stride) {
			if (sm[tid + stride] < sm[tid] ||
				(sm[tid] == sm[tid + stride] && im[tid + stride] < im[tid])) {
				im[tid] = im[tid + stride];
				sm[tid] = sm[tid + stride];
			}
		}
		__syncthreads();
	}
	if (tid == 0) {
		min[blockIdx.x] = sm[0];
		pos[blockIdx.x] = im[0];
	}
}


__global__
void add2overlapmap(int totalLines, int lineLength, int mapInitIdx, int nfpInitIdx, int mapOffsetHeight, int nfpOffsetHeight, quint32* map, quint32* nfp, int weight)
{
	int localRowIdx = blockIdx.y * blockDim.x + threadIdx.x;
	int mapidx = mapInitIdx + blockIdx.x * (lineLength + mapOffsetHeight) + localRowIdx;
	int nfpidx = nfpInitIdx + blockIdx.x * (lineLength + nfpOffsetHeight) + localRowIdx;
	if (localRowIdx < lineLength)
		map[mapidx] += weight * nfp[nfpidx];
}

TotalOverlapMapCuda::TotalOverlapMapCuda(std::shared_ptr<RasterNoFitPolygon> ifp, int _cuttingStockLength) : TotalOverlapMap(ifp, _cuttingStockLength) {
	delete[] data;
	initCuda(width, height);
}

TotalOverlapMapCuda::TotalOverlapMapCuda(QRect& boundingBox, int _cuttingStockLength) : TotalOverlapMap(boundingBox, _cuttingStockLength) {
	delete[] data;
	initCuda(width, height);
}

TotalOverlapMapCuda::TotalOverlapMapCuda(int width, int height, QPoint _reference, int _cuttingStockLength) : TotalOverlapMap(width, height, _reference, _cuttingStockLength) {
	delete[] data;
	initCuda(width, height);
}

TotalOverlapMapCuda::~TotalOverlapMapCuda() {
	cudaFree(data);
	data = nullptr;
}

void TotalOverlapMapCuda::initCuda(uint _width, uint _height) {
	cudaMalloc((void**)&data, _width * _height * sizeof(quint32));
	cudaDeviceSynchronize();
	auto error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("CUDA error allocating total overlap map of size %f MB: %s\n", (float)(_width * _height * sizeof(quint32)) / 1024.0, cudaGetErrorString(error));
		// show memory usage of GPU
		size_t free_byte, total_byte;
		auto cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
		if (cudaSuccess != cuda_status) printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
		else {
			double free_db = (double)free_byte;
			double total_db = (double)total_byte;
			double used_db = total_db - free_db;
			printf("Memory report:: used = %.2f MB, free = %.2f MB, total = %.2f MB\n", used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
		}
	}
	else cudaMemset(data, 0, _width * _height * sizeof(quint32));
}

void TotalOverlapMapCuda::setDimensions(int _width, int _height) {
	cudaFree(data);
	initCuda(_width, _height);
	this->width = _width; this->height = _height;
}

void TotalOverlapMapCuda::reset() {
	cudaMemset(data, 0, width * height * sizeof(quint32));
}

void TotalOverlapMapCuda::addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos) {
	//// Get intersection between innerfit and nofit polygon bounding boxes
	//QPoint relativeOrigin = this->reference + pos - nfp->getOrigin();
	//int relativeBotttomLeftX = relativeOrigin.x() < 0 ? -relativeOrigin.x() : 0;
	//int relativeBotttomLeftY = relativeOrigin.y() < 0 ? -relativeOrigin.y() : 0;
	//int relativeTopRightX = width - relativeOrigin.x(); relativeTopRightX = relativeTopRightX <  nfp->width() ? relativeTopRightX - 1 : nfp->width() - 1;
	//int relativeTopRightY = height - relativeOrigin.y(); relativeTopRightY = relativeTopRightY < nfp->height() ? relativeTopRightY - 1 : nfp->height() - 1;

	//// Create pointers to initial positions and calculate offsets for moving vertically
	//int offsetHeight = height - (relativeTopRightY - relativeBotttomLeftY + 1);
	//int nfpOffsetHeight = nfp->height() - (relativeTopRightY - relativeBotttomLeftY + 1);
	//int mapInitIdx = itemId * width * height + (relativeBotttomLeftX + relativeOrigin.x())*height + relativeBotttomLeftY + relativeOrigin.y();
	//int nfpInitIdx = relativeBotttomLeftY + relativeBotttomLeftX * nfp->height();

	//int totalLines = relativeTopRightX - relativeBotttomLeftX + 1;
	//int lineLength = relativeTopRightY - relativeBotttomLeftY + 1;
	//filloverlapmatrix << < totalLines, THREADS_PER_BLOCK >> >(totalLines, lineLength, mapInitIdx, nfpInitIdx, offsetHeight, nfpOffsetHeight, data, nfp->getPixelRef(0, 0));
}

void TotalOverlapMapCuda::addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, int weight) {
	// Get intersection between innerfit and nofit polygon bounding boxes
	QPoint relativeOrigin = this->reference + pos - nfp->getFlipMultiplier() * nfp->getOrigin() + QPoint(nfp->width() - 1, nfp->height() - 1) * (nfp->getFlipMultiplier() - 1) / 2;
	int relativeBotttomLeftX = relativeOrigin.x() < 0 ? -relativeOrigin.x() : 0;
	int relativeBotttomLeftY = relativeOrigin.y() < 0 ? -relativeOrigin.y() : 0;
	int relativeTopRightX = width - relativeOrigin.x(); relativeTopRightX = relativeTopRightX < nfp->width() ? relativeTopRightX - 1 : nfp->width() - 1;
	int relativeTopRightY = height - relativeOrigin.y(); relativeTopRightY = relativeTopRightY < nfp->height() ? relativeTopRightY - 1 : nfp->height() - 1;

	// Create pointers to initial positions and calculate offsets for moving vertically
	int offsetHeight = height - (relativeTopRightY - relativeBotttomLeftY + 1);
	int nfpOffsetHeight = nfp->height() - (relativeTopRightY - relativeBotttomLeftY + 1);
	int mapInitIdx = (relativeBotttomLeftX + relativeOrigin.x()) * height + relativeBotttomLeftY + relativeOrigin.y();
	int nfpInitIdx = relativeBotttomLeftY + relativeBotttomLeftX * nfp->height();

	int totalLines = relativeTopRightX - relativeBotttomLeftX + 1;
	int lineLength = relativeTopRightY - relativeBotttomLeftY + 1;
	dim3 numBlocks(totalLines, (lineLength + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
	add2overlapmap << < numBlocks, THREADS_PER_BLOCK >> > (totalLines, lineLength, mapInitIdx, nfpInitIdx, offsetHeight, nfpOffsetHeight, data, nfp->getPixelRef(0, 0), weight);
}

quint32 TotalOverlapMapCuda::getMinimum(QPoint& minPt) {
	// TODO: Process more than one stock
	quint32* d_min, * d_pos;
	quint32 maxVal = std::numeric_limits<quint32>::max();
	int n = width * height;
	int numBlocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	cudaMalloc((void**)&d_min, numBlocks * sizeof(quint32));
	cudaMalloc((void**)&d_pos, numBlocks * sizeof(quint32));
	
	findMinimumDevice << < numBlocks, THREADS_PER_BLOCK, 2 * THREADS_PER_BLOCK * sizeof(quint32) >> > (data, n, maxVal, d_min, d_pos);
	int totalIts = numBlocks > 1 ? 1 + numBlocks / THREADS_PER_BLOCK : 0;
	for (int i = 0; i < totalIts; i++) {
		n = numBlocks;
		numBlocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		findMinimumDevice2 << < numBlocks, THREADS_PER_BLOCK, 2 * THREADS_PER_BLOCK * sizeof(quint32) >> > (n, maxVal, d_min, d_pos);
	}
	quint32 minVal, minid;
	cudaMemcpy(&minVal, d_min, sizeof(quint32), cudaMemcpyDeviceToHost);
	cudaMemcpy(&minid, d_pos, sizeof(quint32), cudaMemcpyDeviceToHost);

	minPt = QPoint(minid / height, minid % height) - this->reference;
	return minVal;
}