#include "rasternofitpolygon.h"
#include "../packingProblem.h"
#include<QImage>
#include<QPoint>
using namespace RASTERVORONOIPACKING;

RasterNoFitPolygonSet::RasterNoFitPolygonSet() {
    numAngles = 4;
}

RasterNoFitPolygonSet::RasterNoFitPolygonSet(int numberOfOrientations) : numAngles(numberOfOrientations) {}

void RasterNoFitPolygonSet::addRasterNoFitPolygon(int staticPieceTypeId, int staticAngleId, int orbitingPieceTypeId, int orbitingAngleId, std::shared_ptr<RasterNoFitPolygon> nfp) {
    int staticKey =  staticPieceTypeId*numAngles + staticAngleId;
    int orbitingKey =  orbitingPieceTypeId*numAngles + orbitingAngleId;
    Nfps.insert(QPair<int,int>(staticKey, orbitingKey), nfp);
}

std::shared_ptr<RasterNoFitPolygon> RasterNoFitPolygonSet::getRasterNoFitPolygon(int staticPieceTypeId, int staticAngleId, int orbitingPieceTypeId, int orbitingAngleId) {
    int staticKey =  staticPieceTypeId*numAngles + staticAngleId;
    int orbitingKey =  orbitingPieceTypeId*numAngles + orbitingAngleId;
    return Nfps[QPair<int,int>(staticKey, orbitingKey)];
}

void RasterNoFitPolygonSet::eraseRasterNoFitPolygon(int staticPieceTypeId, int staticAngleId, int orbitingPieceTypeId, int orbitingAngleId) {
    int staticKey =  staticPieceTypeId*numAngles + staticAngleId;
    int orbitingKey =  orbitingPieceTypeId*numAngles + orbitingAngleId;
    Nfps.remove(QPair<int,int>(staticKey, orbitingKey));
}

void RasterNoFitPolygonSet::clear() {
    Nfps.clear();
}
