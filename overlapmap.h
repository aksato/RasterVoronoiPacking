#ifndef OVERLAPMAP_H
#define OVERLAPMAP_H

#include <QtGlobal>
#include <QPoint>
#include "raster/rasternofitpolygon.h"
#include <memory>

namespace RASTERPREPROCESSING {
    class PackingProblem;
    class Polygon;
}


class OverlapMap
{
public:
    OverlapMap();
    OverlapMap(uint _width, uint _height);
    OverlapMap(uint _width, uint _height, QPoint _ref);
    ~OverlapMap();
    void init(uint _width, uint _height);
    quint16 *scanLine(int y);
    void reset();
    std::shared_ptr<RASTERVORONOIPACKING::RasterNoFitPolygon> getImage();

    void setReferencePoint(QPoint _ref) {reference = _ref;}
    QPoint getReferencePoint() {return reference;}
    QPoint getMinimum(quint16 &minVal);
    int getWidth() {return width;}
    int getHeight() {return height;}
    quint16 getValue(int i, int j) {return data[j*width+i];}
    quint16 getValue(QPoint &pt) {return getValue(pt.x(),pt.y());}

    static std::shared_ptr<OverlapMap> createOverlapMap(std::shared_ptr<RASTERPREPROCESSING::Polygon> container, std::shared_ptr<RASTERPREPROCESSING::Polygon> piece, qreal scale = 1.0);
    static std::shared_ptr<OverlapMap> createOverlapMap(std::shared_ptr<RASTERPREPROCESSING::Polygon> container, QRectF pieceBB, qreal scale=1.0);
    void addVoronoi(std::shared_ptr<RASTERVORONOIPACKING::RasterNoFitPolygon> vm, QPoint pos);

private:
    quint16 *data;
    int width;
    int height;
    QPoint reference;
};

class OverlapMapSet {
public:
    OverlapMapSet();

    bool load(RASTERPACKING::PackingProblem &problem);
    void addOverlapMap(int orbitingPieceId, int orbitingAngleId, std::shared_ptr<OverlapMap> ovm);
    std::shared_ptr<OverlapMap> getOverlapMap(int orbitingPieceId, int orbitingAngleId);

private:
    QHash<int, std::shared_ptr<OverlapMap>> ovMaps;
    int numAngles;
};

#endif // OVERLAPMAP_H
