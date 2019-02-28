#include "totaloverlapmap.h"
#include <stdio.h>
using namespace RASTERVORONOIPACKING;

#define THREADS_PER_BLOCK 512

__global__
void filloverlapmatrix(int totalLines, int lineLength, int mapInitIdx, int nfpInitIdx, int mapOffsetHeight, int nfpOffsetHeight, quint32 *map, quint32 *nfp)
{
	int mapidx = mapInitIdx + blockIdx.x * (lineLength + mapOffsetHeight) + threadIdx.x;
	int nfpidx = nfpInitIdx + blockIdx.x * (lineLength + nfpOffsetHeight) + threadIdx.x;
	if (threadIdx.x < lineLength)
		map[mapidx] = nfp[nfpidx];
}

TotalOverlapMapCuda::TotalOverlapMapCuda(std::shared_ptr<RasterNoFitPolygon> ifp, int _numItems, int _cuttingStockLength) : TotalOverlapMap(ifp, _cuttingStockLength), numItems(_numItems) {
	delete[] data;
	initCuda(width, height);
}

TotalOverlapMapCuda::TotalOverlapMapCuda(QRect &boundingBox, int _numItems, int _cuttingStockLength) : TotalOverlapMap(boundingBox, _cuttingStockLength), numItems(_numItems) {
	delete[] data;
	initCuda(width, height);
}

TotalOverlapMapCuda::TotalOverlapMapCuda(int width, int height, QPoint _reference, int _numItems, int _cuttingStockLength) : TotalOverlapMap(width, height, _reference, _cuttingStockLength), numItems(_numItems) {
	delete[] data;
	initCuda(width, height);
}

TotalOverlapMapCuda::~TotalOverlapMapCuda() {
	cudaFree(data);
	data = nullptr;
}

void TotalOverlapMapCuda::initCuda(uint _width, uint _height) {
	cudaFree(data);
	cudaMalloc((void **)&data, _width * _height * numItems * sizeof(quint32));
	cudaMemset(data, 0, _width * _height * numItems * sizeof(quint32));
}

void TotalOverlapMapCuda::setDimensions(int _width, int _height) {
	initCuda(_width, _height);
	this->width = _width; this->height = _height;
}

void TotalOverlapMapCuda::reset(){
	cudaMemset(data, 0, width * height * numItems * sizeof(quint32));
}

void TotalOverlapMapCuda::addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos) {
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
	filloverlapmatrix << < totalLines, THREADS_PER_BLOCK >> >(totalLines, lineLength, mapInitIdx, nfpInitIdx, offsetHeight, nfpOffsetHeight, data, nfp->getPixelRef(0, 0));
}

void TotalOverlapMap::addToMatrixCuda(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, std::shared_ptr<quint32> d_overlapMatrix) {
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
	filloverlapmatrix << < totalLines, THREADS_PER_BLOCK >> >(totalLines, lineLength, mapInitIdx, nfpInitIdx, offsetHeight, nfpOffsetHeight, d_overlapMatrix.get(), nfp->getPixelRef(0,0));
}