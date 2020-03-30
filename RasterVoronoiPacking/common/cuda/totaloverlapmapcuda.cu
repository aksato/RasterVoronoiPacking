#include "totaloverlapmapcuda.h"
using namespace RASTERVORONOIPACKING;

#define THREADS_PER_BLOCK 512

__device__ void getMin(quint32& minVal, quint32& minPos, quint32 val2, quint32 pos2) {
	if (val2 < minVal ||
		(val2 == minVal && pos2 < minPos)) {
		minVal = val2;
		minPos = pos2;
	}
}

// TODO: More efficient parallel minimum algorithm
__global__ void findMinimumDevice(quint32* d, int n, quint32 maxVal, quint32* min, quint32* pos) {
	extern __shared__ quint32 s[];
	quint32* sm = s;
	quint32* im = &sm[blockDim.x];

	uint tid = threadIdx.x;
	uint i = blockIdx.x * (blockDim.x*2) + threadIdx.x;
	if(i >= n)
		sm[tid] = maxVal;
	else if (i + blockDim.x >= n) {
		sm[tid] = d[i];
		im[tid] = i;
	}
	else {
		if (d[i + blockDim.x] < d[i])  {
			sm[tid] = d[i + blockDim.x];
			im[tid] = i + blockDim.x;
		}
		else {
			sm[tid] = d[i];
			im[tid] = i;
		}
	}
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
		if (tid < stride)
			getMin(sm[tid], im[tid], sm[tid + stride], im[tid + stride]);
		__syncthreads();
	}
	if (tid < 32)
	{
		getMin(sm[tid], im[tid], sm[tid + 32], im[tid + 32]);
		getMin(sm[tid], im[tid], sm[tid + 16], im[tid + 16]);
		getMin(sm[tid], im[tid], sm[tid + 8], im[tid + 8]);
		getMin(sm[tid], im[tid], sm[tid + 4], im[tid + 4]);
		getMin(sm[tid], im[tid], sm[tid + 2], im[tid + 2]);
		getMin(sm[tid], im[tid], sm[tid + 1], im[tid + 1]);
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
	uint i = blockIdx.x * (blockDim.x*2) + threadIdx.x;

	if (i >= n)
		sm[tid] = maxVal;
	else if (i + blockDim.x >= n) {
		sm[tid] = min[i];
		im[tid] = pos[i];
	}
	else {
		if (min[i + blockDim.x] < min[i] ||
			(min[i] == min[tid + blockDim.x] && pos[i + blockDim.x] < pos[i])) {
			sm[tid] = min[i + blockDim.x];
			im[tid] = pos[i + blockDim.x];
		}
		else {
			sm[tid] = min[i];
			im[tid] = pos[i];
		}
	}
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
		if (tid < stride)
			getMin(sm[tid], im[tid], sm[tid + stride], im[tid + stride]);
		__syncthreads();
	}
	if (tid < 32)
	{
		getMin(sm[tid], im[tid], sm[tid + 32], im[tid + 32]);
		getMin(sm[tid], im[tid], sm[tid + 16], im[tid + 16]);
		getMin(sm[tid], im[tid], sm[tid + 8], im[tid + 8]);
		getMin(sm[tid], im[tid], sm[tid + 4], im[tid + 4]);
		getMin(sm[tid], im[tid], sm[tid + 2], im[tid + 2]);
		getMin(sm[tid], im[tid], sm[tid + 1], im[tid + 1]);
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
	Q_ASSERT_X(_width > 0 && _height > 0, "TotalOverlapMap::shrink", "Item does not fit the container");
	if (_width > this->width || _height > this->height) {
		// Expanding the map buffer
		cudaFree(data);
		initCuda(_width, _height);
	}
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
	int numBlocks = (n + (THREADS_PER_BLOCK * 2 - 1)) / (THREADS_PER_BLOCK * 2);
	cudaMalloc((void**)&d_min, numBlocks * sizeof(quint32));
	cudaMalloc((void**)&d_pos, numBlocks * sizeof(quint32));
	
	findMinimumDevice << < numBlocks, THREADS_PER_BLOCK, 2 * THREADS_PER_BLOCK * sizeof(quint32) >> > (data, n, maxVal, d_min, d_pos);
	int totalIts = numBlocks > 1 ? 1 + numBlocks / THREADS_PER_BLOCK : 0;
	for (int i = 0; i < totalIts; i++) {
		n = numBlocks;
		numBlocks = (n + (THREADS_PER_BLOCK * 2 - 1)) / (THREADS_PER_BLOCK * 2);
		findMinimumDevice2 << < numBlocks, THREADS_PER_BLOCK, 2 * THREADS_PER_BLOCK * sizeof(quint32) >> > (n, maxVal, d_min, d_pos);
	}
	quint32 minVal, minid;
	cudaMemcpy(&minVal, d_min, sizeof(quint32), cudaMemcpyDeviceToHost);
	cudaMemcpy(&minid, d_pos, sizeof(quint32), cudaMemcpyDeviceToHost);

	minPt = QPoint(minid / height, minid % height) - this->reference;
	return minVal;
}