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
//    init(ifp->getImage().width(), ifp->getImage().height());
    init(ifp->width(), ifp->height());
    reference = ifp->getOrigin();
}

TotalOverlapMap::TotalOverlapMap(int width, int height) : originalWidth(width), originalHeight(height) {
    init(width, height);
}

TotalOverlapMap::TotalOverlapMap(QRect &boundingBox) : originalWidth(boundingBox.width()), originalHeight(boundingBox.height()) {
	init(boundingBox.width(), boundingBox.height());
	reference = -boundingBox.topLeft();
}

void TotalOverlapMap::init(uint _width, uint _height) {
    this->width = _width;
    this->height = _height;
    data = new float[width*height];
    Q_CHECK_PTR(data);
    std::fill(data, data+width*height, (float)0.0);
    #ifdef QT_DEBUG
        initialWidth = width;
    #endif
}

void TotalOverlapMap::reset() {
     std::fill(data, data+width*height, (float)0.0);
}

float *TotalOverlapMap::scanLine(int y) {
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

void TotalOverlapMap::addVoronoi(std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos) {
    QPoint relativeOrigin = getReferencePoint() + pos - nfp->getOrigin();
    QRect intersection;
//    if(!getLimits(relativeOrigin, nfp->getImage().width(), nfp->getImage().height(), intersection)) return;
    if(!getLimits(relativeOrigin, nfp->width(), nfp->height(), intersection)) return;

    for(int i = intersection.bottomLeft().x(); i <= intersection.topRight().x(); i++)
        for(int j = intersection.bottomLeft().y(); j <= intersection.topRight().y(); j++) {
            int indexValue = nfp->getPixel(i-relativeOrigin.x(), j-relativeOrigin.y());
//            int indexValue = nfp->getImage().pixelIndex(i-relativeOrigin.x(), j-relativeOrigin.y());
            float distanceValue = 0.0;
            //if(indexValue != 0) distanceValue = 1.0 + (nfp->getMaxD()-1.0)*((float)indexValue-1.0)/254.0;
            //data[j*width+i] += distanceValue;
			data[j*width + i] += indexValue;
//            data[j*width+i] += nfp->getImage().pixelIndex(i-relativeOrigin.x(), j-relativeOrigin.y());
        }

}

void TotalOverlapMap::addVoronoi(std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, float weight) {
    QPoint relativeOrigin = getReferencePoint() + pos - nfp->getOrigin();
    QRect intersection;
//    if(!getLimits(relativeOrigin, nfp->getImage().width(), nfp->getImage().height(), intersection)) return;
    if(!getLimits(relativeOrigin, nfp->width(), nfp->height(), intersection)) return;

    for(int i = intersection.bottomLeft().x(); i <= intersection.topRight().x(); i++)
        for(int j = intersection.bottomLeft().y(); j <= intersection.topRight().y(); j++) {
            int indexValue = nfp->getPixel(i-relativeOrigin.x(), j-relativeOrigin.y());
//            int indexValue = nfp->getImage().pixelIndex(i-relativeOrigin.x(), j-relativeOrigin.y());
            float distanceValue = 0.0;
            //if(indexValue != 0) distanceValue = 1.0 + (nfp->getMaxD()-1.0)*((float)indexValue-1.0)/254.0;
            //data[j*width+i] += weight*distanceValue;
			data[j*width + i] += weight*(float)indexValue;
        }
}

//void TotalOverlapMap::setZoomedMap(QPoint nonZoomedpos, float zoomFactor) {
//    QPoint bottomLeft = QPoint(qRound(nonZoomedpos.x()*zoomFactor), qRound(nonZoomedpos.y()*zoomFactor));
//    for(int j = bottomLeft.y; j < bottomLeft.y+width; j++)
//        for(int i = bottomLeft.x; i < bottomLeft.x+width; i++)

//}

QPoint TotalOverlapMap::getMinimum(float &minVal, PositionChoice placementHeuristic) {
	QVector<QPoint> minPointSet;
    float *curPt = data;
    minVal = *curPt;
	minPointSet.push_back(QPoint(0,0));
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++, curPt++) {
			float curVal = *curPt;
			if (qFuzzyCompare(1.0 + curVal, 1.0 + minVal)) {
				minPointSet.push_back(QPoint(i, j));
			}
			else if (curVal < minVal) {
				minVal = curVal;
				minPointSet.clear(); minPointSet.push_back(QPoint(i, j));
			}
		}
	}

	// Heuristics to choose between equally overlaped positions
	int index = 0;
	switch(placementHeuristic) {
		case BOTTOMLEFT_POS:
		{
		   index = 0;
		}
		break;

		case RANDOM_POS: 
		{
			index = rand() % minPointSet.size();
		}
		break;

		case LIMITS_POS:
		{
			int blIndex, lbIndex, tlIndex, ltIndex;
			int brIndex, rbIndex, trIndex, rtIndex;
			blIndex = 0; lbIndex = 0; tlIndex = 0; ltIndex = 0;
			brIndex = 0; rbIndex = 0; trIndex = 0; rtIndex = 0;
			for (int i = 1; i < minPointSet.size(); i++) {
				QPoint curPoint = minPointSet[i];
				if (curPoint.y() < minPointSet[blIndex].y() || (curPoint.y() == minPointSet[blIndex].y() && curPoint.x() < minPointSet[blIndex].x())) blIndex = i;
				if (curPoint.x() < minPointSet[lbIndex].x() || (curPoint.x() == minPointSet[lbIndex].x() && curPoint.y() < minPointSet[lbIndex].y())) lbIndex = i;
				if (curPoint.y() > minPointSet[tlIndex].y() || (curPoint.y() == minPointSet[tlIndex].y() && curPoint.x() < minPointSet[tlIndex].x())) tlIndex = i;
				if (curPoint.x() < minPointSet[ltIndex].x() || (curPoint.x() == minPointSet[ltIndex].x() && curPoint.y() > minPointSet[ltIndex].y())) ltIndex = i;

				if (curPoint.y() < minPointSet[brIndex].y() || (curPoint.y() == minPointSet[brIndex].y() && curPoint.x() > minPointSet[brIndex].x())) brIndex = i;
				if (curPoint.x() > minPointSet[rbIndex].x() || (curPoint.x() == minPointSet[rbIndex].x() && curPoint.y() < minPointSet[rbIndex].y())) rbIndex = i;
				if (curPoint.y() > minPointSet[trIndex].y() || (curPoint.y() == minPointSet[trIndex].y() && curPoint.x() > minPointSet[trIndex].x())) trIndex = i;
				if (curPoint.x() > minPointSet[rtIndex].x() || (curPoint.x() == minPointSet[rtIndex].x() && curPoint.y() > minPointSet[rtIndex].y())) rtIndex = i;
			}
			int d8 = rand() % 8;
			if (d8 == 0) index = blIndex;
			else if (d8 == 1) index = lbIndex;
			else if (d8 == 2) index = tlIndex;
			else if (d8 == 3) index = ltIndex;
			else if (d8 == 4) index = brIndex;
			else if (d8 == 5) index = rbIndex;
			else if (d8 == 6) index = trIndex;
			else if (d8 == 7) index = rtIndex;
		}
		break;

		case CONTOUR_POS:
		{
			QVector<int> contourPtsIndex;
			for (int i = 0; i < minPointSet.size(); i++) {
				QPoint curPoint = minPointSet[i];
				// Check if it is in the contour
				if (curPoint.x() == 0 || curPoint.x() == width - 1 || curPoint.y() == 0 || curPoint.y() == height - 1) contourPtsIndex.push_back(i);
				else if (!qFuzzyCompare(1.0 + getLocalValue(curPoint.x() - 1, curPoint.y() - 1), 1.0 + minVal)) contourPtsIndex.push_back(i);
				else if (!qFuzzyCompare(1.0 + getLocalValue(curPoint.x(), curPoint.y() - 1), 1.0 + minVal)) contourPtsIndex.push_back(i);
				else if (!qFuzzyCompare(1.0 + getLocalValue(curPoint.x() + 1, curPoint.y() - 1), 1.0 + minVal)) contourPtsIndex.push_back(i);
				else if (!qFuzzyCompare(1.0 + getLocalValue(curPoint.x() - 1, curPoint.y()), 1.0 + minVal)) contourPtsIndex.push_back(i);
				else if (!qFuzzyCompare(1.0 + getLocalValue(curPoint.x() + 1, curPoint.y()), 1.0 + minVal)) contourPtsIndex.push_back(i);
				else if (!qFuzzyCompare(1.0 + getLocalValue(curPoint.x() - 1, curPoint.y() + 1), 1.0 + minVal)) contourPtsIndex.push_back(i);
				else if (!qFuzzyCompare(1.0 + getLocalValue(curPoint.x(), curPoint.y() + 1), 1.0 + minVal)) contourPtsIndex.push_back(i);
				else if (!qFuzzyCompare(1.0 + getLocalValue(curPoint.x() + 1, curPoint.y() + 1), 1.0 + minVal)) contourPtsIndex.push_back(i);
			}
			//for (int i = 0; i < contourPtsIndex.size(); i++) qDebug() << minPointSet[contourPtsIndex[i]];
			index = contourPtsIndex[rand() % contourPtsIndex.size()];
		}
		break;
	}

	return minPointSet[index];
}

#ifndef CONSOLE
    QImage TotalOverlapMap::getImage() {
        float maxD = 0;
        for(int pixelY = 0; pixelY < height; pixelY++) {
            float *mapLine = scanLine(pixelY);
            for(int pixelX = 0; pixelX < width; pixelX++) {
                if(*mapLine > maxD) maxD = *mapLine;
                mapLine++;
            }
        }

        QImage image(width, height, QImage::Format_Indexed8);
        setColormap(image);
        for(int pixelY = 0; pixelY < height; pixelY++) {
            uchar *resultLine = (uchar *)image.scanLine(pixelY);
            float *mapLine = scanLine(pixelY);
            for(int pixelX = 0; pixelX < width; pixelX++) {
                if(*mapLine==0)
                    *resultLine=0;
                else {
                    int index = (int)((*mapLine-1)*254/(maxD-1) + 1);
                    *resultLine = index;
                }
                resultLine ++; mapLine++;
            }
        }
        return image;
    }
#endif

//QImage TotalOverlapMap::getImage2() {
//    float maxD = 0;
//    float minD = data[0];
//    for(int pixelY = 0; pixelY < height; pixelY++) {
//        float *mapLine = scanLine(pixelY);
//        for(int pixelX = 0; pixelX < width; pixelX++) {
//            if(*mapLine > maxD) maxD = *mapLine;
//            if(!qFuzzyCompare(1.0 + 0.0, 1.0 + minD) && *mapLine < minD) minD = *mapLine;
//            mapLine++;
//        }
//    }

//    QImage image(width, height, QImage::Format_Indexed8);
//    setColormap(image);
//    for(int pixelY = 0; pixelY < height; pixelY++) {
//        uchar *resultLine = (uchar *)image.scanLine(pixelY);
//        float *mapLine = scanLine(pixelY);
//        for(int pixelX = 0; pixelX < width; pixelX++) {
//            if(qFuzzyCompare(1.0 + 0.0, 1.0 + *mapLine))
//                *resultLine=0;
//            else {
//                int index = (int)((*mapLine-minD)*254/(maxD-minD) + 1);
//                *resultLine = index;
//            }
//            resultLine ++; mapLine++;
//        }
//    }
//    return image;
//}
