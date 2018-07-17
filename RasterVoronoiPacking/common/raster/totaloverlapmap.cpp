#include "totaloverlapmap.h"

#ifndef CONSOLE
	#include "colormap.h"
#endif

using namespace RASTERVORONOIPACKING;

TotalOverlapMapSet::TotalOverlapMapSet(int numItems) : shrinkValX(0), shrinkValY(0) {
    numAngles = 4;
	initializeSet(numItems);
}

TotalOverlapMapSet::TotalOverlapMapSet(int numberOfOrientations, int numItems) : shrinkValX(0), shrinkValY(0), numAngles(numberOfOrientations) {
	initializeSet(numItems);
}

void TotalOverlapMapSet::initializeSet(int numItems) {
	mapSet = QVector<std::shared_ptr<TotalOverlapMap>>(numItems*numAngles);
}

void TotalOverlapMapSet::addOverlapMap(int orbitingPieceId, int orbitingAngleId, std::shared_ptr<TotalOverlapMap> ovm) {
    int orbitingKey =  orbitingPieceId*numAngles + orbitingAngleId;
    //mapSet.insert(orbitingKey, ovm);
	mapSet[orbitingKey] = ovm;
}

std::shared_ptr<TotalOverlapMap> TotalOverlapMapSet::getOverlapMap(int orbitingPieceId, int orbitingAngleId) {
    int orbitingKey =  orbitingPieceId*numAngles + orbitingAngleId;
    return mapSet[orbitingKey];
}

TotalOverlapMap::TotalOverlapMap(std::shared_ptr<RasterNoFitPolygon> ifp, int _cuttingStockLength) : originalWidth(ifp->width()), originalHeight(ifp->height()), cuttingStockLength(_cuttingStockLength) {
	zoomFactor = 1;
	initialWidth = -1;
	reference = ifp->getOrigin();
    init(ifp->width(), ifp->height());
}

TotalOverlapMap::TotalOverlapMap(int width, int height, QPoint _reference, int _cuttingStockLength) : originalWidth(width), originalHeight(height), reference(_reference), cuttingStockLength(_cuttingStockLength) {
	zoomFactor = 1;
	initialWidth = -1;
	init(width, height);
}

TotalOverlapMap::TotalOverlapMap(QRect &boundingBox, int _cuttingStockLength) : originalWidth(boundingBox.width()), originalHeight(boundingBox.height()), cuttingStockLength(_cuttingStockLength) {
	zoomFactor = 1;
	initialWidth = -1;
	reference = -boundingBox.topLeft();
	init(boundingBox.width(), boundingBox.height());
}

void TotalOverlapMap::init(uint _width, uint _height) {
    this->width = _width;
    this->height = _height;
	data = new quint32[width*height];
    Q_CHECK_PTR(data);
	//#ifdef QT_DEBUG
    if(initialWidth == -1) initialWidth = width;
    //#endif
	reset(); //std::fill(data, data+width*height, 0);
}

void TotalOverlapMap::reset() {
	std::fill(data, data+width*height, 0);
}

quint32 *TotalOverlapMap::scanLine(int x) {
	return data+x*height;
}

bool TotalOverlapMap::getLimits(QPoint relativeOrigin, int vmWidth, int vmHeight, QRect &intersection) {
    intersection.setCoords(relativeOrigin.x() < 0 ? 0 : relativeOrigin.x(),
                           relativeOrigin.y()+vmHeight > getHeight() ? getHeight()-1 : relativeOrigin.y()+vmHeight-1,
                           relativeOrigin.x()+vmWidth > getWidth() ? getWidth()-1 : relativeOrigin.x()+vmWidth-1,
                           relativeOrigin.y() < 0 ? 0 : relativeOrigin.y());

    if(intersection.topRight().x() < 0 || intersection.topRight().y() < 0 ||
       intersection.bottomLeft().x() > getWidth() || intersection.bottomLeft().y() > getHeight()) return false;
    return true;
}

bool TotalOverlapMap::getLimits(QPoint relativeOrigin, int vmWidth, int vmHeight, QRect &intersection, int zoomFactorInt) {
	intersection.setCoords(relativeOrigin.x() < 0 ? 0 : relativeOrigin.x(),
		relativeOrigin.y() + vmHeight > zoomFactorInt*getHeight() ? zoomFactorInt*getHeight() - 1 : relativeOrigin.y() + vmHeight - 1,
		relativeOrigin.x() + vmWidth > zoomFactorInt*getWidth() ? zoomFactorInt*getWidth() - 1 : relativeOrigin.x() + vmWidth - 1,
		relativeOrigin.y() < 0 ? 0 : relativeOrigin.y());

	if (intersection.topRight().x() < 0 || intersection.topRight().y() < 0 ||
		intersection.bottomLeft().x() > zoomFactorInt*getWidth() || intersection.bottomLeft().y() > zoomFactorInt*getHeight()) return false;
	return true;
}

void TotalOverlapMap::addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos) {
	// Get intersection between innerfit and nofit polygon bounding boxes
	QPoint relativeOrigin = this->reference + pos - nfp->getOrigin();
	int relativeBotttomLeftX = relativeOrigin.x() < 0 ? -relativeOrigin.x() : 0;
	int relativeBotttomLeftY = relativeOrigin.y() < 0 ? -relativeOrigin.y() : 0;
	int relativeTopRightX = width - relativeOrigin.x(); relativeTopRightX = relativeTopRightX <  nfp->width() ? relativeTopRightX - 1 : nfp->width() - 1;
	int relativeTopRightY = height - relativeOrigin.y(); relativeTopRightY = relativeTopRightY < nfp->height() ? relativeTopRightY - 1 : nfp->height() - 1;

	// Create pointers to initial positions and calculate offsets for moving vertically
	int offsetHeight = height - (relativeTopRightY - relativeBotttomLeftY + 1);
	int nfpOffsetHeight = nfp->height() - (relativeTopRightY - relativeBotttomLeftY + 1);
	quint32 *mapPointer = scanLine(relativeBotttomLeftX + relativeOrigin.x()) + relativeBotttomLeftY + relativeOrigin.y();
	quint32 *nfpPointer = nfp->getPixelRef(relativeBotttomLeftX, relativeBotttomLeftY);

	// Add nofit polygon values to overlap map
	for (int i = relativeBotttomLeftX; i <= relativeTopRightX; i++) {
		for (int j = relativeBotttomLeftY; j <= relativeTopRightY; j++, mapPointer++, nfpPointer += nfp->getFlipMultiplier())
			*mapPointer += *nfpPointer;
		mapPointer += offsetHeight; nfpPointer += nfp->getFlipMultiplier()*nfpOffsetHeight;
	}
}

void TotalOverlapMap::addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, int weight) {
	// Get intersection between innerfit and nofit polygon bounding boxes
	QPoint relativeOrigin = this->reference + pos - nfp->getFlipMultiplier()*nfp->getOrigin() + QPoint(nfp->width() - 1, nfp->height() - 1)*(nfp->getFlipMultiplier() - 1) / 2;
	//QPoint relativeOrigin = this->reference + pos - nfp->getOrigin();
	//QPoint relativeOrigin = -this->reference - QPoint(nfp->width() - 1, nfp->height() - 1) + pos;
	int relativeBotttomLeftX = relativeOrigin.x() < 0 ? -relativeOrigin.x() : 0;
	int relativeBotttomLeftY = relativeOrigin.y() < 0 ? -relativeOrigin.y() : 0;
	int relativeTopRightX =  width - relativeOrigin.x(); relativeTopRightX = relativeTopRightX <  nfp->width() ? relativeTopRightX - 1 :  nfp->width() - 1;
	int relativeTopRightY = height - relativeOrigin.y(); relativeTopRightY = relativeTopRightY < nfp->height() ? relativeTopRightY - 1 : nfp->height() - 1;

	// Create pointers to initial positions and calculate offsets for moving vertically
	int offsetHeight = height - (relativeTopRightY - relativeBotttomLeftY + 1);
	int nfpOffsetHeight = nfp->height() - (relativeTopRightY - relativeBotttomLeftY + 1);
	quint32 *mapPointer = scanLine(relativeBotttomLeftX + relativeOrigin.x()) + relativeBotttomLeftY + relativeOrigin.y();
	quint32 *nfpPointer = nfp->getPixelRef(relativeBotttomLeftX, relativeBotttomLeftY);

	// Add nofit polygon values to overlap map
	for (int i = relativeBotttomLeftX; i <= relativeTopRightX; i++) {
		for (int j = relativeBotttomLeftY; j <= relativeTopRightY; j++, mapPointer++, nfpPointer += nfp->getFlipMultiplier())
			*mapPointer += weight * (*nfpPointer);
		mapPointer += offsetHeight; nfpPointer += nfp->getFlipMultiplier()*nfpOffsetHeight;
	}
}

void TotalOverlapMap::addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, int weight, int zoomFactorInt) {
	// Get intersection between innerfit and nofit polygon bounding boxes
	QPoint relativeOrigin = zoomFactorInt * this->reference + pos - nfp->getFlipMultiplier()*nfp->getOrigin() + QPoint(nfp->width() - 1, nfp->height() - 1)*(nfp->getFlipMultiplier() - 1) / 2;
	//QPoint relativeOrigin = zoomFactorInt * this->reference + pos - nfp->getOrigin();
	int relativeBotttomLeftX = relativeOrigin.x() < 0 ? 0 : (relativeOrigin.x() % zoomFactorInt == 0 ? relativeOrigin.x() / zoomFactorInt : relativeOrigin.x() / zoomFactorInt + 1);
	int relativeBotttomLeftY = relativeOrigin.y() < 0 ? 0 : (relativeOrigin.y() % zoomFactorInt == 0 ? relativeOrigin.y() / zoomFactorInt : relativeOrigin.y() / zoomFactorInt + 1);
	int relativeWidth = (relativeOrigin.x() + nfp->width() - 1) / zoomFactorInt; if (relativeWidth >= width) relativeWidth = width - relativeBotttomLeftX; else relativeWidth = relativeWidth - relativeBotttomLeftX + 1;
	int relativeHeight = (relativeOrigin.y() + nfp->height() - 1) / zoomFactorInt; if (relativeHeight >= height) relativeHeight = height - relativeBotttomLeftY; else relativeHeight = relativeHeight - relativeBotttomLeftY + 1;

	// Create pointers to initial positions and calculate offsets for moving vertically
	int offsetHeight = height - relativeHeight;
	int nfpOffsetHeight = zoomFactorInt*nfp->height() - zoomFactorInt*relativeHeight;
	quint32 *mapPointer = scanLine(relativeBotttomLeftX) + relativeBotttomLeftY;
	quint32 *nfpPointer = nfp->getPixelRef(zoomFactorInt * relativeBotttomLeftX - relativeOrigin.x(), zoomFactorInt * relativeBotttomLeftY - relativeOrigin.y());

	// Add nofit polygon values to overlap map
	for (int i = 0; i < relativeWidth; i++) {
		for (int j = 0; j < relativeHeight; j++, mapPointer++, nfpPointer += nfp->getFlipMultiplier()*zoomFactorInt)
			*mapPointer += weight * (*nfpPointer);
		mapPointer += offsetHeight; nfpPointer += nfp->getFlipMultiplier()*nfpOffsetHeight;
	}
}

quint32 TotalOverlapMap::getMinimum(QPoint &minPt) {
	quint32 minVal = std::numeric_limits<quint32>::max();
	int minid = 0;
	int id = 0;
	if (cuttingStockLength < 0) findMinimum(minVal, minid, id, height * width); 
	else {
		while (id < height*width) {
			findMinimum(minVal, minid, id, height * initialWidth);

			//findMinimum(minVal, minid, id, curPt, height * width);
			if (minVal == 0) break;
			id += (cuttingStockLength - initialWidth) * height;
		}
	}
	minPt = QPoint(minid / height, minid % height) - this->reference;
	return minVal;
}

quint32 TotalOverlapMap::getMinimum(QPoint &minPt, int &stockLocation) {
	quint32 minVal = std::numeric_limits<quint32>::max();
	int minid = 0;
	int id = 0;
	int currentStock = 0;
	if (cuttingStockLength < 0) { findMinimum(minVal, minid, id, height * width); stockLocation = -1;}
	else {
		while (id < height*width) {
			//findMinimum(minVal, minid, id, height * initialWidth);
			int lastVal = std::min(id + height * initialWidth, height*width);
			quint32 *curPt = &data[id];
			for (; id < lastVal; id++, curPt++) {
				quint32 curVal = *curPt;
				//if (curVal < minVal || minVal == 0) {
				if (curVal < minVal) {
					minVal = curVal;
					minid = id;
					stockLocation = currentStock;
					if (minVal == 0) break;
				}
			}

			//findMinimum(minVal, minid, id, curPt, height * width);
			if (minVal == 0) break;
			id += (cuttingStockLength - initialWidth) * height; currentStock++;
		}
	}
	minPt = QPoint(minid / height, minid % height) - this->reference;
	return minVal;
}
void TotalOverlapMap::findMinimum(quint32 &minVal, int &minid, int &curid, int blockSize) {
	//int lastVal = std::min(height * initialWidth, height*width);
	int lastVal = std::min(curid + blockSize, height*width);
	quint32 *curPt = &data[curid];
	for (; curid < lastVal; curid++, curPt++) {
		quint32 curVal = *curPt;
		//if (curVal < minVal || minVal == 0) {
		if (curVal < minVal) {
			minVal = curVal;
			minid = curid;
			if (minVal == 0) break;
		}
	}
}

quint32 TotalOverlapMap::getBottomLeft(QPoint &minPt, bool borderOk) {
	quint32 *curPt = data;
	quint32 minVal = std::numeric_limits<quint32>::max();
	int minid = 0;
	//minPt = QPoint(0, 0);
	int numVals = height*width;

	for (int index = (borderOk ? 0 : height); index < (borderOk ? numVals : numVals - height); index++, curPt++) {
		if (!borderOk && (index % height == 0 || index % (height - 1) == 0)) continue;
		quint32 curVal = *curPt;
		if (curVal < minVal || minVal == 0) {
			minVal = curVal;
			minid = index;
			if (minVal == 0) break;
		}
	}
	minPt = QPoint(minid / height, minid % height) - this->reference;
	return minVal;
}

void TotalOverlapMap::setZoomFactor(int _zoomFactor) {
	if (this->zoomFactor > 1) return; // Cannot set zoom factor more than once
	this->zoomFactor = _zoomFactor;
	this->cuttingStockLength /= this->zoomFactor;
}

#ifndef CONSOLE
    QImage TotalOverlapMap::getImage() {
		float maxD = 0;
		for (int pixelX = 0; pixelX < width; pixelX++) {
			quint32 *mapLine = scanLine(pixelX);
			for (int pixelY = 0; pixelY < height; pixelY++) {
                if(*mapLine > maxD) maxD = *mapLine;
                mapLine++;
            }
        }

        QImage image(width, height, QImage::Format_Indexed8);
        setColormap(image);
        for(int pixelY = 0; pixelY < height; pixelY++) {
            uchar *resultLine = (uchar *)image.scanLine(pixelY);
            for(int pixelX = 0; pixelX < width; pixelX++) {
				quint32 *mapLine = data + pixelX*height + pixelY;
                if(*mapLine==0)
                    *resultLine=0;
                else {
                    int index = (int)(((float)*mapLine-1.0)*254.0/(maxD-1.0) + 1.0);
                    *resultLine = index;
                }
                resultLine ++;
            }
        }
        return image;
    }

	QImage TotalOverlapMap::getZoomImage(int _width, int _height, QPoint &displacement) {
		float maxD = 0;
		for (int pixelX = 0; pixelX < width; pixelX++) {
			quint32 *mapLine = scanLine(pixelX);
			for (int pixelY = 0; pixelY < height; pixelY++) {
				if (*mapLine > maxD) maxD = *mapLine;
				mapLine++;
			}
		}

		QImage image(_width, _height, QImage::Format_Indexed8);
		image.fill(255);
		setColormap(image);
		image.setColor(255, qRgba(0, 0, 0, 0));
		for (int pixelY = 0; pixelY < _height; pixelY++) {
			uchar *resultLine = (uchar *)image.scanLine(pixelY);
			int mapPixelY = pixelY - displacement.y(); if (mapPixelY < 0 || mapPixelY >= height) continue;

			int pixelX = 0;
			int mapPixelX = pixelX - displacement.x();
			while (mapPixelX < 0) resultLine++, mapPixelX++, pixelX++;
			for (; pixelX < _width; pixelX++, resultLine++, mapPixelX++) {
				quint32 *mapLine = data + mapPixelX*height + mapPixelY;
				if (mapPixelX >= width) break;
				if (*mapLine == 0)
					*resultLine = 0;
				else {
					int index = (int)(((float)*mapLine - 1.0)*253.0 / (maxD - 1.0) + 1.0);
					*resultLine = index;
				}
			}
		}
		return image;
	}

	void TotalOverlapMap::maskCuttingStock() {
		int id = 0;
		if (cuttingStockLength < 0) return;
		quint32 maxVal = 0;
		std::vector<int> maskVec(height*width, 0);
		while (id < height*width) {
			//findMinimum(minVal, minid, id, height * initialWidth);
			int blockSize = height * initialWidth;
			int lastVal = std::min(id + blockSize, height*width);
			for (quint32 *curPt = &data[id]; id < lastVal; id++, curPt++) {
				maskVec[id] = 1;
				if (*curPt > maxVal) maxVal = *curPt;
			}
			//for (int i = 0; i < height * initialWidth; i++) maskVec[id + i] = 1; id += height * initialWidth;

			id += (cuttingStockLength - initialWidth) * height;
		}
		for (int i = 0; i < height * width; i++) data[i] = maskVec[i] == 1 ? data[i] : maxVal;
	}
#endif