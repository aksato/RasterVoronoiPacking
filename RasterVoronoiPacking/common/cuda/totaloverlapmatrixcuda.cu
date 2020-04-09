#include "cuda/totaloverlapmatrixcuda.h"
using namespace RASTERVORONOIPACKING;

#define THREADS_PER_BLOCK 512

__global__
void filloverlapmatrix(int totalLines, int lineLength, int mapInitIdx, int nfpInitIdx, int mapOffsetHeight, int nfpOffsetHeight, quint32 *map, quint32 *nfp, int multiplier)
{
	int mapidx = mapInitIdx + blockIdx.x * (lineLength + mapOffsetHeight) + blockIdx.y * blockDim.y + threadIdx.x;
	int nfpidx = multiplier * (nfpInitIdx + blockIdx.x * (lineLength + nfpOffsetHeight) + blockIdx.y * blockDim.y + threadIdx.x);
	if (threadIdx.x < lineLength)
		map[mapidx] = nfp[nfpidx];
}

TotalOverlapMatrixCuda::TotalOverlapMatrixCuda(std::shared_ptr<RasterNoFitPolygon> ifp, int _numItems, std::vector<cudaStream_t> &_streams, int _cuttingStockLength) : TotalOverlapMap(ifp, _cuttingStockLength), numItems(_numItems), streams(_streams) {
	delete[] data;
	initCuda(width, height);
}

TotalOverlapMatrixCuda::TotalOverlapMatrixCuda(QRect &boundingBox, int _numItems, std::vector<cudaStream_t> &_streams, int _cuttingStockLength) : TotalOverlapMap(boundingBox, _cuttingStockLength), numItems(_numItems), streams(_streams) {
	delete[] data;
	initCuda(width, height);
}

TotalOverlapMatrixCuda::TotalOverlapMatrixCuda(int width, int height, QPoint _reference, int _numItems, std::vector<cudaStream_t> &_streams, int _cuttingStockLength) : TotalOverlapMap(width, height, _reference, _cuttingStockLength), numItems(_numItems), streams(_streams) {
	delete[] data;
	initCuda(width, height);
}

TotalOverlapMatrixCuda::~TotalOverlapMatrixCuda() {
	cudaFree(data);
	data = nullptr;
}

void TotalOverlapMatrixCuda::initCuda(uint _width, uint _height) {
	cudaMalloc((void **)&data, _width * _height * numItems * sizeof(quint32));
	cudaDeviceSynchronize();
	auto error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("CUDA error allocating total overlap matrix of size %f MB: %s\n", (float)(_width * _height * sizeof(quint32)) / 1024.0, cudaGetErrorString(error));
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
	else cudaMemset(data, 0, _width * _height * numItems * sizeof(quint32));
}

void TotalOverlapMatrixCuda::setDimensions(int _width, int _height) {
	cudaFree(data);
	initCuda(_width, _height);
	this->width = _width; this->height = _height;
}

void TotalOverlapMatrixCuda::reset(){
	for(int itemId = 0; itemId < numItems; itemId++)
		cudaMemsetAsync(data + width * height * itemId, 0, width * height * sizeof(quint32), streams[itemId]);
}

void TotalOverlapMatrixCuda::addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos) {
	// Get intersection between innerfit and nofit polygon bounding boxes
	QPoint relativeOrigin = this->reference + pos - nfp->getFlipMultiplier() * nfp->getOrigin() + QPoint(nfp->width() - 1, nfp->height() - 1) * (nfp->getFlipMultiplier() - 1) / 2;
	int relativeBotttomLeftX = relativeOrigin.x() < 0 ? -relativeOrigin.x() : 0;
	int relativeBotttomLeftY = relativeOrigin.y() < 0 ? -relativeOrigin.y() : 0;
	int relativeTopRightX = width - relativeOrigin.x(); relativeTopRightX = relativeTopRightX < nfp->width() ? relativeTopRightX - 1 : nfp->width() - 1;
	int relativeTopRightY = height - relativeOrigin.y(); relativeTopRightY = relativeTopRightY < nfp->height() ? relativeTopRightY - 1 : nfp->height() - 1;

	// Create pointers to initial positions and calculate offsets for moving vertically
	int offsetHeight = height - (relativeTopRightY - relativeBotttomLeftY + 1);
	int nfpOffsetHeight = nfp->height() - (relativeTopRightY - relativeBotttomLeftY + 1);
	int mapInitIdx = itemId * width * height + (relativeBotttomLeftX + relativeOrigin.x())*height + relativeBotttomLeftY + relativeOrigin.y();
	int nfpInitIdx = relativeBotttomLeftY + relativeBotttomLeftX * nfp->height();

	int totalLines = relativeTopRightX - relativeBotttomLeftX + 1;
	int lineLength = relativeTopRightY - relativeBotttomLeftY + 1;
	dim3 numBlocks(totalLines, (lineLength + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
	filloverlapmatrix << < numBlocks, THREADS_PER_BLOCK, 0, streams[itemId] >> >(totalLines, lineLength, mapInitIdx, nfpInitIdx, offsetHeight, nfpOffsetHeight, data, nfp->getPixelRef(0, 0), nfp->getFlipMultiplier());
}