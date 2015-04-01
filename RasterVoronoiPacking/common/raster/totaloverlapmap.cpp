#include "totaloverlapmap.h"

#ifndef CONSOLE
	#include "colormap.h"
#endif

using namespace RASTERVORONOIPACKING;

TotalOverlapMapSet::TotalOverlapMapSet(){
    numAngles = 4;
}

TotalOverlapMapSet::TotalOverlapMapSet(int numberOfOrientations) : numAngles(numberOfOrientations) {}

void TotalOverlapMapSet::addOverlapMap(int orbitingPieceId, int orbitingAngleId, std::shared_ptr<TotalOverlapMap> ovm) {
    int orbitingKey =  orbitingPieceId*numAngles + orbitingAngleId;
    mapSet.insert(orbitingKey, ovm);
}

std::shared_ptr<TotalOverlapMap> TotalOverlapMapSet::getOverlapMap(int orbitingPieceId, int orbitingAngleId) {
    int orbitingKey =  orbitingPieceId*numAngles + orbitingAngleId;
    return mapSet[orbitingKey];
}

TotalOverlapMap::TotalOverlapMap(std::shared_ptr<RasterNoFitPolygon> ifp) {
//    init(ifp->getImage().width(), ifp->getImage().height());
    init(ifp->width(), ifp->height());
    reference = ifp->getOrigin();
}

TotalOverlapMap::TotalOverlapMap(int width, int height) {
    init(width, height);
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

QPoint TotalOverlapMap::getMinimum(float &minVal) {
    float *curPt = data;
    minVal = *curPt;
    QPoint minPoint;
    minPoint.setX(0); minPoint.setY(0);
    for(int j = 0; j < height; j++)
        for(int i = 0; i < width; i++,curPt++) {
            float curVal = *curPt;
            if(curVal < minVal) {
                minVal = curVal;
                minPoint.setX(i); minPoint.setY(j);
            }
        }

    return minPoint;
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
