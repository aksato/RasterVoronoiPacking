#include "totaloverlapmap.h"

#ifndef CONSOLE
	#include "colormap.h"
#endif

using namespace RASTERVORONOIPACKING;

TotalOverlapMapSet::TotalOverlapMapSet() : shrinkValX(0), shrinkValY(0) {
    numAngles = 4;
}

TotalOverlapMapSet::TotalOverlapMapSet(int numberOfOrientations) : shrinkValX(0), shrinkValY(0), numAngles(numberOfOrientations) {}

void TotalOverlapMapSet::addOverlapMap(int orbitingPieceId, int orbitingAngleId, std::shared_ptr<TotalOverlapMap> ovm) {
    int orbitingKey =  orbitingPieceId*numAngles + orbitingAngleId;
    mapSet.insert(orbitingKey, ovm);
}

std::shared_ptr<TotalOverlapMap> TotalOverlapMapSet::getOverlapMap(int orbitingPieceId, int orbitingAngleId) {
    int orbitingKey =  orbitingPieceId*numAngles + orbitingAngleId;
    return mapSet[orbitingKey];
}

TotalOverlapMap::TotalOverlapMap(std::shared_ptr<RasterNoFitPolygon> ifp) : originalWidth(ifp->width()), originalHeight(ifp->height()) {
    init(ifp->width(), ifp->height());
    reference = ifp->getOrigin();
}

TotalOverlapMap::TotalOverlapMap(int width, int height) : originalWidth(width), originalHeight(height) {
    init(width, height);
}

TotalOverlapMap::TotalOverlapMap(int width, int height, QPoint _reference) : originalWidth(width), originalHeight(height), reference(_reference) {
	init(width, height);
}

TotalOverlapMap::TotalOverlapMap(QRect &boundingBox) : originalWidth(boundingBox.width()), originalHeight(boundingBox.height()) {
	init(boundingBox.width(), boundingBox.height());
	reference = -boundingBox.topLeft();
}

void TotalOverlapMap::init(uint _width, uint _height) {
    this->width = _width;
    this->height = _height;
	data = new quint32[width*height];
    Q_CHECK_PTR(data);
    std::fill(data, data+width*height, 0);
    #ifdef QT_DEBUG
        initialWidth = width;
    #endif
}

void TotalOverlapMap::reset() {
     std::fill(data, data+width*height, 0);
}

quint32 *TotalOverlapMap::scanLine(int y) {
    return data+y*width;
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

void TotalOverlapMap::addVoronoi(std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos) {
    QPoint relativeOrigin = getReferencePoint() + pos - nfp->getOrigin();
    QRect intersection;
    if(!getLimits(relativeOrigin, nfp->width(), nfp->height(), intersection)) return;

	for (int j = intersection.bottomLeft().y(); j <= intersection.topRight().y(); j++) {
		quint32 *dataPt = scanLine(j) + intersection.bottomLeft().x();
		for (int i = intersection.bottomLeft().x(); i <= intersection.topRight().x(); i++, dataPt++) {
			*dataPt += nfp->getPixel(i - relativeOrigin.x(), j - relativeOrigin.y());
		}
	}
}

void TotalOverlapMap::addVoronoi(std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, int weight) {
    QPoint relativeOrigin = getReferencePoint() + pos - nfp->getOrigin();
    QRect intersection;
    if(!getLimits(relativeOrigin, nfp->width(), nfp->height(), intersection)) return;

	for (int j = intersection.bottomLeft().y(); j <= intersection.topRight().y(); j++) {
		quint32 *dataPt = scanLine(j) + intersection.bottomLeft().x();
		for (int i = intersection.bottomLeft().x(); i <= intersection.topRight().x(); i++, dataPt++) {
			*dataPt += weight * nfp->getPixel(i - relativeOrigin.x(), j - relativeOrigin.y());
		}
	}
}

void TotalOverlapMap::addVoronoi(std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, int weight, int zoomFactorInt) {
	//QPoint relativeOrigin = getReferencePoint() + pos - nfp->getOrigin();
	QPoint relativeOrigin = zoomFactorInt * getReferencePoint() + pos - nfp->getOrigin();
	QRect intersection;
	if (!getLimits(relativeOrigin, nfp->width(), nfp->height(), intersection, zoomFactorInt)) return;

	int initialY = intersection.bottomLeft().y() % zoomFactorInt == 0 ? intersection.bottomLeft().y() : zoomFactorInt * ((intersection.bottomLeft().y() / zoomFactorInt) + 1);
	for (int j = initialY; j <= intersection.topRight().y(); j += zoomFactorInt) {
		int initialX = intersection.bottomLeft().x() % zoomFactorInt == 0 ? intersection.bottomLeft().x() : zoomFactorInt * ((intersection.bottomLeft().x() / zoomFactorInt) + 1);
		quint32 *dataPt = scanLine(j / 5) + initialX / 5;
		for (int i = initialX; i <= intersection.topRight().x(); i += zoomFactorInt, dataPt++) {
			quint32 indexValue = nfp->getPixel(i - relativeOrigin.x(), j - relativeOrigin.y());
			*dataPt += weight * indexValue;
		}
	}
}

quint32 TotalOverlapMap::getMinimum(QPoint &minPt) {
	quint32 *curPt = data;
	quint32 minVal = *curPt;
	int minid = 0;
	minPt = QPoint(0, 0);
	int numVals = height*width;
	for (int id = 0; id < numVals; id++, curPt++) {
		quint32 curVal = *curPt;
		if (curVal < minVal) {
			minVal = curVal;
			minid = id;
			if (minVal == 0) {
				minPt = QPoint(minid % width, minid / width);
				return minVal;
			}
		}
	}
	minPt = QPoint(minid % width, minid / width);
	return minVal;
}

#ifndef CONSOLE
    QImage TotalOverlapMap::getImage() {
		float maxD = 0;
        for(int pixelY = 0; pixelY < height; pixelY++) {
            quint32 *mapLine = scanLine(pixelY);
            for(int pixelX = 0; pixelX < width; pixelX++) {
                if(*mapLine > maxD) maxD = *mapLine;
                mapLine++;
            }
        }

        QImage image(width, height, QImage::Format_Indexed8);
        setColormap(image);
        for(int pixelY = 0; pixelY < height; pixelY++) {
            uchar *resultLine = (uchar *)image.scanLine(pixelY);
			quint32 *mapLine = scanLine(pixelY);
            for(int pixelX = 0; pixelX < width; pixelX++) {
                if(*mapLine==0)
                    *resultLine=0;
                else {
                    int index = (int)(((float)*mapLine-1.0)*254.0/(maxD-1.0) + 1.0);
                    *resultLine = index;
                }
                resultLine ++; mapLine++;
            }
        }
        return image;
    }

	QImage TotalOverlapMap::getZoomImage(int _width, int _height, QPoint &displacement) {
		float maxD = 0;
		for (int pixelY = 0; pixelY < height; pixelY++) {
			quint32 *mapLine = scanLine(pixelY);
			for (int pixelX = 0; pixelX < width; pixelX++) {
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
			quint32 *mapLine = scanLine(mapPixelY);

			int pixelX = 0;
			int mapPixelX = pixelX - displacement.x();
			while (mapPixelX < 0) resultLine++, mapPixelX++, pixelX++;
			for (; pixelX < _width; pixelX++, resultLine++, mapLine++, mapPixelX++) {
				if (mapPixelX >= width) break;
				if (*mapLine == 0)
					*resultLine = 0;
				else {
					int index = (int)(((float)*mapLine - 1.0)*254.0 / (maxD - 1.0) + 1.0);
					*resultLine = index;
				}
			}
		}
		return image;
	}
#endif