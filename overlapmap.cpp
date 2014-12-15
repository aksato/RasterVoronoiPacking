#include "overlapmap.h"
#include "raster/rasternofitpolygon.h"
#include <QImage>
#include <memory>
#include "packingProblem.h"
#include "colormap.h"

using namespace RASTERVORONOIPACKING;

OverlapMapSet::OverlapMapSet() {
    numAngles = 4;
}

void OverlapMapSet::addOverlapMap(int orbitingPieceId, int orbitingAngleId, std::shared_ptr<OverlapMap> ovm) {
    int orbitingKey =  orbitingPieceId*numAngles + orbitingAngleId;
    ovMaps.insert(orbitingKey, ovm);
}

std::shared_ptr<OverlapMap> OverlapMapSet::getOverlapMap(int orbitingPieceId, int orbitingAngleId) {
    int orbitingKey =  orbitingPieceId*numAngles + orbitingAngleId;
    return ovMaps[orbitingKey];
}

std::shared_ptr<OverlapMap> OverlapMap::createOverlapMap(std::shared_ptr<RASTERPREPROCESSING::Polygon> container, QRectF pieceBB, qreal scale) {
    qreal pieceMinX, pieceMinY, pieceW, pieceH, containerW, containerH;
    pieceMinX = scale*pieceBB.bottomLeft().x(); pieceMinY = scale*pieceBB.topRight().y();
    pieceW = scale*pieceBB.width(); pieceH = scale*pieceBB.height();

    QPolygonF containerQPolF = *container;
    QRectF containerBB = containerQPolF.boundingRect();
    containerW = scale*containerBB.width(); containerH = scale*containerBB.height();

    int iwidth = qRound(containerW - pieceW)+1;
    int iheight = qRound(containerH - pieceH)+1;
    QPoint RP(qRound(pieceMinX), qRound(pieceMinY));

    return std::shared_ptr<OverlapMap>(new OverlapMap(iwidth, iheight, RP));
}

std::shared_ptr<OverlapMap> OverlapMap::createOverlapMap(std::shared_ptr<RASTERPREPROCESSING::Polygon> container, std::shared_ptr<RASTERPREPROCESSING::Polygon> piece, qreal scale) {
    qreal pieceMinX, pieceMinY, pieceW, pieceH, containerW, containerH;
    QPolygonF pieceQPolF = *piece;
    QRectF pieceBB = pieceQPolF.boundingRect();
    pieceMinX = scale*pieceBB.bottomLeft().x(); pieceMinY = scale*pieceBB.topRight().y();
    pieceW = scale*pieceBB.width(); pieceH = scale*pieceBB.height();

    QPolygonF containerQPolF = *container;
    QRectF containerBB = containerQPolF.boundingRect();
    containerW = scale*containerBB.width(); containerH = scale*containerBB.height();

    int iwidth = qRound(containerW - pieceW)+1;
    int iheight = qRound(containerH - pieceH)+1;
    QPoint RP(qRound(pieceMinX), qRound(pieceMinY));

    return std::shared_ptr<OverlapMap>(new OverlapMap(iwidth, iheight, RP));
}

//void OverlapMap::resetOverlapMap(std::shared_ptr<RASTERPACKING::Polygon> container, std::shared_ptr<RASTERPACKING::Polygon> piece, qreal scale = 1.0) {
//    qreal pieceMinX, pieceMinY, pieceW, pieceH, containerW, containerH;
//    QPolygonF pieceQPolF = *piece;
//    QRectF pieceBB = pieceQPolF.boundingRect();
//    pieceMinX = scale*pieceBB.bottomLeft().x(); pieceMinY = scale*pieceBB.topRight().y();
//    pieceW = scale*pieceBB.width(); pieceH = scale*pieceBB.height();

//    QPolygonF containerQPolF = *container;
//    QRectF containerBB = containerQPolF.boundingRect();
//    containerW = scale*containerBB.width(); containerH = scale*containerBB.height();

//    int iwidth = qRound(containerW - pieceW)+1;
//    int iheight = qRound(containerH - pieceH)+1;
//    QPoint RP(qRound(pieceMinX), qRound(pieceMinY));

//    init(iwidth, iheight);
//    reference = RP;
//}

OverlapMap::OverlapMap()
{
}

OverlapMap::OverlapMap(uint _width, uint _height) {
    init(_width, _height);
}

OverlapMap::OverlapMap(uint _width, uint _height, QPoint _ref) {
    init(_width, _height);
    reference = _ref;
}

OverlapMap::~OverlapMap() {
    delete[] data;
}

void OverlapMap::init(uint _width, uint _height) {
    this->width = _width;
    this->height = _height;
    data = new quint16[width*height];
    std::fill(data, data+width*height, 0);
}

quint16 *OverlapMap::scanLine(int y) {
    return data+y*width;
}

std::shared_ptr<RasterNoFitPolygon> OverlapMap::getImage() {
    quint16 maxD = 0;
    for(int pixelY = 0; pixelY < height; pixelY++) {
        quint16 *mapLine = scanLine(pixelY);
        for(int pixelX = 0; pixelX < width; pixelX++) {
            if(*mapLine > maxD) maxD = *mapLine;
            mapLine++;
        }
    }

    QImage image(width, height, QImage::Format_Indexed8);
    setColormap(image);
    for(int pixelY = 0; pixelY < height; pixelY++) {
        uchar *resultLine = (uchar *)image.scanLine(pixelY);
        quint16 *mapLine = scanLine(pixelY);
        for(int pixelX = 0; pixelX < width; pixelX++) {
            if(*mapLine==0)
                *resultLine=0;
            else {
                int index = (int)((*mapLine-1)*254/(maxD-1) + 1);
                *resultLine = index;//qRound(255*((float)*mapLine/(float)maxD));
            }
            resultLine ++; mapLine++;
        }
    }
    return std::shared_ptr<RasterNoFitPolygon>(new RasterNoFitPolygon(image, reference, (qreal)maxD));
}

QPoint OverlapMap::getMinimum(quint16 &minVal) {
    quint16 *curPt = data;
    minVal = *curPt;
    QPoint minPoint;
    minPoint.setX(0); minPoint.setY(0);
    for(int j = 0; j < height; j++)
        for(int i = 0; i < width; i++,curPt++) {
            quint16 curVal = *curPt;
            if(curVal < minVal) {
                minVal = curVal;
                minPoint.setX(i); minPoint.setY(j);
//                if(minVal == 0) {qDebug() << "Zero!";break;}
            }
        }

    return minPoint;
}

void OverlapMap::reset() {
     std::fill(data, data+width*height, 0);
}

void OverlapMap::addVoronoi(std::shared_ptr<RasterNoFitPolygon> vm, QPoint pos) {
    QPoint relativeOrigin = getReferencePoint() + pos - vm->getOrigin();

    QRect ovRect, vmRect;
    int ovXmin = relativeOrigin.x() < 0 ? 0 : relativeOrigin.x();
    int ovXmax = relativeOrigin.x()+vm->getImage().width() > getWidth() ? getWidth()-1 : relativeOrigin.x()+vm->getImage().width()-1;
    int ovYmin = relativeOrigin.y() < 0 ? 0 : relativeOrigin.y();
    int ovYmax = relativeOrigin.y()+vm->getImage().height() > getHeight() ? getHeight()-1 : relativeOrigin.y()+vm->getImage().height()-1;

    if(ovXmax < 0 || ovYmax < 0 || ovXmin > getWidth() || ovYmin > getHeight()) return;

    for(int i = ovXmin; i <= ovXmax; i++)
        for(int j = ovYmin; j <= ovYmax; j++)
            data[j*width+i] += vm->getImage().pixelIndex(i-relativeOrigin.x(), j-relativeOrigin.y());

}
