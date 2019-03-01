#include "totaloverlapmatrix.h"
#include <stdio.h>
using namespace RASTERVORONOIPACKING;

TotalOverlapMatrix::TotalOverlapMatrix(std::shared_ptr<RasterNoFitPolygon> ifp, int _numItems, int _cuttingStockLength) : TotalOverlapMap(ifp, _cuttingStockLength), numItems(_numItems) {
	delete[] data;
	initMatrix(width, height);
}

TotalOverlapMatrix::TotalOverlapMatrix(QRect &boundingBox, int _numItems, int _cuttingStockLength) : TotalOverlapMap(boundingBox, _cuttingStockLength), numItems(_numItems) {
	delete[] data;
	initMatrix(width, height);
}

TotalOverlapMatrix::TotalOverlapMatrix(int width, int height, QPoint _reference, int _numItems, int _cuttingStockLength) : TotalOverlapMap(width, height, _reference, _cuttingStockLength), numItems(_numItems) {
	delete[] data;
	initMatrix(width, height);
}
void TotalOverlapMatrix::initMatrix(uint _width, uint _height) {
	matrixData = Eigen::Matrix< unsigned int, Eigen::Dynamic, Eigen::Dynamic >::Zero(_width * _height, numItems);
}

void TotalOverlapMatrix::setDimensions(int _width, int _height) {
	initMatrix(_width, _height);
	this->width = _width; this->height = _height;
}

void TotalOverlapMatrix::reset(){
	initMatrix(width, height);
}

void TotalOverlapMatrix::addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos) {
	// Get intersection between innerfit and nofit polygon bounding boxes
	QPoint relativeOrigin = this->reference + pos - nfp->getOrigin();
	int relativeBotttomLeftX = relativeOrigin.x() < 0 ? -relativeOrigin.x() : 0;
	int relativeBotttomLeftY = relativeOrigin.y() < 0 ? -relativeOrigin.y() : 0;
	int relativeTopRightX = width - relativeOrigin.x(); relativeTopRightX = relativeTopRightX <  nfp->width() ? relativeTopRightX - 1 : nfp->width() - 1;
	int relativeTopRightY = height - relativeOrigin.y(); relativeTopRightY = relativeTopRightY < nfp->height() ? relativeTopRightY - 1 : nfp->height() - 1;

	// Create pointers to initial positions and calculate offsets for moving vertically
	int offsetHeight = height - (relativeTopRightY - relativeBotttomLeftY + 1);
	int nfpOffsetHeight = nfp->height() - (relativeTopRightY - relativeBotttomLeftY + 1);
	quint32 *mapPointer = matrixData.data() + itemId * matrixData.rows() + (relativeBotttomLeftX + relativeOrigin.x())*height + relativeBotttomLeftY + relativeOrigin.y();
	quint32 *nfpPointer = nfp->getPixelRef(relativeBotttomLeftX, relativeBotttomLeftY);

	// Add nofit polygon values to overlap map
	for (int i = relativeBotttomLeftX; i <= relativeTopRightX; i++) {
		int lineLength = relativeTopRightY - relativeBotttomLeftY + 1;
		std::memcpy(mapPointer, nfpPointer, lineLength * sizeof(quint32));
		mapPointer += lineLength + offsetHeight; nfpPointer += lineLength + nfpOffsetHeight;
	}
}