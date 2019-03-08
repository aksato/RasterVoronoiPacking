#include "totaloverlapmapcuda.h"
#include <stdio.h>
using namespace RASTERVORONOIPACKING;

#define THREADS_PER_BLOCK 512

__global__
void add2overlapmap(int totalLines, int lineLength, int mapInitIdx, int nfpInitIdx, int mapOffsetHeight, int nfpOffsetHeight, quint32 *map, quint32 *nfp, int weight)
{
	int mapidx = mapInitIdx + blockIdx.x * (lineLength + mapOffsetHeight) + blockIdx.y * blockDim.y + threadIdx.x;
	int nfpidx = nfpInitIdx + blockIdx.x * (lineLength + nfpOffsetHeight) + blockIdx.y * blockDim.y + threadIdx.x;
	if (threadIdx.x < lineLength)
		map[mapidx] += weight*nfp[nfpidx];
}

TotalOverlapMapCuda::TotalOverlapMapCuda(std::shared_ptr<RasterNoFitPolygon> ifp, int _cuttingStockLength) : TotalOverlapMap(ifp, _cuttingStockLength) {
	delete[] data;
	initCuda(width, height);
}

TotalOverlapMapCuda::TotalOverlapMapCuda(QRect &boundingBox, int _cuttingStockLength) : TotalOverlapMap(boundingBox, _cuttingStockLength) {
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
	cudaMalloc((void **)&data, _width * _height * sizeof(quint32));
	cudaMemset(data, 0, _width * _height * sizeof(quint32));
}

void TotalOverlapMapCuda::setDimensions(int _width, int _height) {
	cudaFree(data);
	initCuda(_width, _height);
	this->width = _width; this->height = _height;
}

void TotalOverlapMapCuda::reset(){
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
	QPoint relativeOrigin = this->reference + pos - nfp->getFlipMultiplier()*nfp->getOrigin() + QPoint(nfp->width() - 1, nfp->height() - 1)*(nfp->getFlipMultiplier() - 1) / 2;
	int relativeBotttomLeftX = relativeOrigin.x() < 0 ? -relativeOrigin.x() : 0;
	int relativeBotttomLeftY = relativeOrigin.y() < 0 ? -relativeOrigin.y() : 0;
	int relativeTopRightX = width - relativeOrigin.x(); relativeTopRightX = relativeTopRightX <  nfp->width() ? relativeTopRightX - 1 : nfp->width() - 1;
	int relativeTopRightY = height - relativeOrigin.y(); relativeTopRightY = relativeTopRightY < nfp->height() ? relativeTopRightY - 1 : nfp->height() - 1;

	// Create pointers to initial positions and calculate offsets for moving vertically
	int offsetHeight = height - (relativeTopRightY - relativeBotttomLeftY + 1);
	int nfpOffsetHeight = nfp->height() - (relativeTopRightY - relativeBotttomLeftY + 1);
	int mapInitIdx = (relativeBotttomLeftX + relativeOrigin.x())*height + relativeBotttomLeftY + relativeOrigin.y();
	int nfpInitIdx = relativeBotttomLeftY + relativeBotttomLeftX * nfp->height();

	int totalLines = relativeTopRightX - relativeBotttomLeftX + 1;
	int lineLength = relativeTopRightY - relativeBotttomLeftY + 1;
	dim3 numBlocks(totalLines, (lineLength + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
	add2overlapmap << < numBlocks, THREADS_PER_BLOCK >> >(totalLines, lineLength, mapInitIdx, nfpInitIdx, offsetHeight, nfpOffsetHeight, data, nfp->getPixelRef(0, 0), weight);
}