#include "rasternofitpolygon.h"
#include "../packingproblem.h"
#include<QImage>
#include<QPoint>
using namespace RASTERVORONOIPACKING;

ItemRasterNoFitPolygonSet::ItemRasterNoFitPolygonSet(int numberOfOrientations, int numItems) : numAngles(numberOfOrientations) {
	itemNfpSet = QVector<std::shared_ptr<RasterNoFitPolygon>>(numItems*numAngles);
}

std::shared_ptr<RasterNoFitPolygon> ItemRasterNoFitPolygonSet::getRasterNoFitPolygon(int staticPieceTypeId, int staticAngleId) {
	int staticKey = staticPieceTypeId*numAngles + staticAngleId;
	return itemNfpSet[staticKey];
}

RasterNoFitPolygonSet::RasterNoFitPolygonSet(int numItems) {
    numAngles = 4;
	initializeSet(numItems);
}

RasterNoFitPolygonSet::RasterNoFitPolygonSet(int numberOfOrientations, int numItems) : numAngles(numberOfOrientations) {
	initializeSet(numItems);
}

void RasterNoFitPolygonSet::initializeSet(int numItems) {
	nfpSet = QVector<std::shared_ptr<ItemRasterNoFitPolygonSet>>(numItems*numAngles);
	for (int i = 0; i < nfpSet.length(); i++) {
		nfpSet[i] = std::shared_ptr<ItemRasterNoFitPolygonSet>(new ItemRasterNoFitPolygonSet(numAngles, numItems));
	}
}

void RasterNoFitPolygonSet::addRasterNoFitPolygon(int staticPieceTypeId, int staticAngleId, int orbitingPieceTypeId, int orbitingAngleId, std::shared_ptr<RasterNoFitPolygon> nfp) {
    int staticKey =  staticPieceTypeId*numAngles + staticAngleId;
    int orbitingKey =  orbitingPieceTypeId*numAngles + orbitingAngleId;
    //Nfps.insert(QPair<int,int>(staticKey, orbitingKey), nfp);
	(*nfpSet[orbitingKey])[staticKey] = nfp;
}

std::shared_ptr<RasterNoFitPolygon> RasterNoFitPolygonSet::getRasterNoFitPolygon(int staticPieceTypeId, int staticAngleId, int orbitingPieceTypeId, int orbitingAngleId) {
    int staticKey =  staticPieceTypeId*numAngles + staticAngleId;
    int orbitingKey =  orbitingPieceTypeId*numAngles + orbitingAngleId;
	return (*nfpSet[orbitingKey])[staticKey];//Nfps[QPair<int,int>(staticKey, orbitingKey)];
}

std::shared_ptr<ItemRasterNoFitPolygonSet> RasterNoFitPolygonSet::getItemRasterNoFitPolygonSet(int orbitingPieceTypeId, int orbitingAngleId) {
	int orbitingKey = orbitingPieceTypeId*numAngles + orbitingAngleId;
	return nfpSet[orbitingKey];
}

//void RasterNoFitPolygonSet::eraseRasterNoFitPolygon(int staticPieceTypeId, int staticAngleId, int orbitingPieceTypeId, int orbitingAngleId) {
//    int staticKey =  staticPieceTypeId*numAngles + staticAngleId;
//    int orbitingKey =  orbitingPieceTypeId*numAngles + orbitingAngleId;
//    Nfps.remove(QPair<int,int>(staticKey, orbitingKey));
//}
//
//void RasterNoFitPolygonSet::clear() {
//    Nfps.clear();
//}

void RasterNoFitPolygon::setMatrix(QImage image) {
    //matrix = std::vector< std::vector<quint8> >(image.height(), std::vector<quint8>(image.width()));
	matrix = new quint32[image.width() * image.height()];
	
    for(int j = 0; j < image.height(); j++)
        for(int i = 0; i < image.width(); i++)
			matrix[j*image.width() + i] = image.pixelIndex(i, j);

	w = image.width(); h = image.height();
//    for(int pixelY = 0; pixelY < image.height(); pixelY++) {
//        uchar *scanLine = (uchar *)image.scanLine(pixelY);
//        for(int pixelX = 0; pixelX < image.width(); pixelX++, scanLine++)
//                matrix[pixelY][pixelX] = *scanLine;
//    }
}
