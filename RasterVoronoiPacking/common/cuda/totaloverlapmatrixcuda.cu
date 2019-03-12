#include "cuda/totaloverlapmatrixcuda.h"
using namespace RASTERVORONOIPACKING;

#define THREADS_PER_BLOCK 512

__global__
void filloverlapmatrix(int totalLines, int lineLength, int mapInitIdx, int nfpInitIdx, int mapOffsetHeight, int nfpOffsetHeight, quint32 *map, quint32 *nfp)
{
	int mapidx = mapInitIdx + blockIdx.x * (lineLength + mapOffsetHeight) + blockIdx.y * blockDim.y + threadIdx.x;
	int nfpidx = nfpInitIdx + blockIdx.x * (lineLength + nfpOffsetHeight) + blockIdx.y * blockDim.y + threadIdx.x;
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
	cudaMemset(data, 0, _width * _height * numItems * sizeof(quint32));
}

void TotalOverlapMatrixCuda::setDimensions(int _width, int _height) {
	cudaFree(data);
	initCuda(_width, _height);
	this->width = _width; this->height = _height;
}

void TotalOverlapMatrixCuda::reset(){
	cudaMemset(data, 0, width * height * numItems * sizeof(quint32));
}

void TotalOverlapMatrixCuda::addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos) {
	// Get intersection between innerfit and nofit polygon bounding boxes
	QPoint relativeOrigin = this->reference + pos - nfp->getOrigin();
	int relativeBotttomLeftX = relativeOrigin.x() < 0 ? -relativeOrigin.x() : 0;
	int relativeBotttomLeftY = relativeOrigin.y() < 0 ? -relativeOrigin.y() : 0;
	int relativeTopRightX = width - relativeOrigin.x(); relativeTopRightX = relativeTopRightX <  nfp->width() ? relativeTopRightX - 1 : nfp->width() - 1;
	int relativeTopRightY = height - relativeOrigin.y(); relativeTopRightY = relativeTopRightY < nfp->height() ? relativeTopRightY - 1 : nfp->height() - 1;

	// Create pointers to initial positions and calculate offsets for moving vertically
	int offsetHeight = height - (relativeTopRightY - relativeBotttomLeftY + 1);
	int nfpOffsetHeight = nfp->height() - (relativeTopRightY - relativeBotttomLeftY + 1);
	int mapInitIdx = itemId * width * height + (relativeBotttomLeftX + relativeOrigin.x())*height + relativeBotttomLeftY + relativeOrigin.y();
	int nfpInitIdx = relativeBotttomLeftY + relativeBotttomLeftX * nfp->height();

	int totalLines = relativeTopRightX - relativeBotttomLeftX + 1;
	int lineLength = relativeTopRightY - relativeBotttomLeftY + 1;
	dim3 numBlocks(totalLines, (lineLength + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
	filloverlapmatrix << < numBlocks, THREADS_PER_BLOCK, 0, streams[itemId] >> >(totalLines, lineLength, mapInitIdx, nfpInitIdx, offsetHeight, nfpOffsetHeight, data, nfp->getPixelRef(0, 0));
}