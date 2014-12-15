#include "rasterpackingproblem.h"
#include "../packingproblem.h"

using namespace RASTERVORONOIPACKING;

RasterPackingProblem::RasterPackingProblem()
{
}

RasterPackingProblem::RasterPackingProblem(RASTERPREPROCESSING::PackingProblem &problem) {
    this->load(problem);
}

void mapPieceNameAngle(RASTERPREPROCESSING::PackingProblem &problem, QHash<QString, int> &pieceIndexMap, QHash<int, int> &pieceTotalAngleMap, QHash<QPair<int,int>, int> &pieceAngleMap) {
    int id = 0;
    for(QList<std::shared_ptr<RASTERPREPROCESSING::Piece>>::const_iterator it = problem.cpbegin(); it != problem.cpend(); it++) {
        pieceIndexMap.insert((*it)->getPolygon()->getName(), id);
        pieceTotalAngleMap.insert(id, (*it)->getOrientationsCount());
        int id2 = 0;
        for(QVector<unsigned int>::const_iterator it2 = (*it)->corbegin(); it2 != (*it)->corend(); it2++) {
            pieceAngleMap.insert(QPair<int,int>(id,id2),*it2);
            id2++;
        }
        id++;
    }
}

QPair<int,int> getIdsFromRasterPreProblem(QString polygonName, int angleValue, QHash<QString, int> &pieceIndexMap, QHash<int, int> &pieceTotalAngleMap, QHash<QPair<int,int>, int> &pieceAngleMap) {
    QPair<int,int> ids;
    ids.first = -1; ids.second = -1;
    if(pieceIndexMap.find(polygonName) != pieceIndexMap.end()) {
        ids.first = *pieceIndexMap.find(polygonName);
//        for(uint i = 0; i < items[ids.first]->getAngleCount(); i++) // ERROR
          for(int i = 0; i < pieceTotalAngleMap[ids.first]; i++)
              if(angleValue == pieceAngleMap[QPair<int,int>(ids.first,i)]) {ids.second = i; break;}
    }
    return ids;
}

bool RasterPackingProblem::load(RASTERPREPROCESSING::PackingProblem &problem) {
    // 1. Load items information
    int typeId = 0; int itemId = 0;
    for(QList<std::shared_ptr<RASTERPREPROCESSING::Piece>>::const_iterator it = problem.cpbegin(); it != problem.cpend(); it++, typeId++)
        for(uint mult = 0; mult < (*it)->getMultiplicity(); mult++, itemId++)
            items.append(std::shared_ptr<RasterPackingItem>(new RasterPackingItem(itemId, typeId, (*it)->getOrientationsCount())));
    std::shared_ptr<RASTERPREPROCESSING::Container> container = *problem.ccbegin();
    std::shared_ptr<RASTERPREPROCESSING::Polygon> pol = container->getPolygon();
    qreal minX = (*pol->begin()).x();
    qreal maxX = minX;
    std::for_each(pol->begin(), pol->end(), [&minX, &maxX](QPointF pt){
        if(pt.x() < minX) minX = pt.x();
        if(pt.x() > maxX) maxX = pt.x();
    });
    qreal rasterScale = (*problem.crnfpbegin())->getScale(); // FIXME: Global scale
    containerWidth = qRound(rasterScale*(maxX - minX));


    // 2. Link the name and angle of the piece with the ids defined for the items
    QHash<QString, int> pieceIndexMap;
    QHash<int, int> pieceTotalAngleMap;
    QHash<QPair<int,int>, int> pieceAngleMap;
    mapPieceNameAngle(problem, pieceIndexMap, pieceTotalAngleMap, pieceAngleMap);

    // 3. Load innerfit polygons
    innerFitPolygons = std::shared_ptr<RasterNoFitPolygonSet>(new RasterNoFitPolygonSet);
    for(QList<std::shared_ptr<RASTERPREPROCESSING::RasterInnerFitPolygon>>::const_iterator it = problem.crifpbegin(); it != problem.crifpend(); it++) {
        std::shared_ptr<RASTERPREPROCESSING::RasterInnerFitPolygon> curRasterIfp = *it;
        // Create image. FIXME: Use data file instead?
        QImage curImage(curRasterIfp->getFileName());
        std::shared_ptr<RasterNoFitPolygon> curMap(new RasterNoFitPolygon(curImage, curRasterIfp->getReferencePoint(), 1.0));

        // Determine ids
        QPair<int,int> staticIds, orbitingIds;
        staticIds = getIdsFromRasterPreProblem(curRasterIfp->getStaticName(), curRasterIfp->getStaticAngle(), pieceIndexMap, pieceTotalAngleMap, pieceAngleMap);
        orbitingIds = getIdsFromRasterPreProblem(curRasterIfp->getOrbitingName(), curRasterIfp->getOrbitingAngle(), pieceIndexMap, pieceTotalAngleMap, pieceAngleMap);
//        if(staticIds.first == -1 || staticIds.second == -1 || orbitingIds.first == -1 || orbitingIds.second == -1) return false;

        // Create nofit polygon
        innerFitPolygons->addRasterNoFitPolygon(staticIds.first, staticIds.second, orbitingIds.first, orbitingIds.second, curMap);
    }

    // 4. Load nofit polygons
    noFitPolygons = std::shared_ptr<RasterNoFitPolygonSet>(new RasterNoFitPolygonSet);
    for(QList<std::shared_ptr<RASTERPREPROCESSING::RasterNoFitPolygon>>::const_iterator it = problem.crnfpbegin(); it != problem.crnfpend(); it++) {
        std::shared_ptr<RASTERPREPROCESSING::RasterNoFitPolygon> curRasterNfp = *it;
        // Create image. FIXME: Use data file instead?
        QImage curImage(curRasterNfp->getFileName());
        std::shared_ptr<RasterNoFitPolygon> curMountain(new RasterNoFitPolygon(curImage, curRasterNfp->getReferencePoint(), curRasterNfp->getMaxD()));

        // Determine ids
        QPair<int,int> staticIds, orbitingIds;
        staticIds = getIdsFromRasterPreProblem(curRasterNfp->getStaticName(), curRasterNfp->getStaticAngle(), pieceIndexMap, pieceTotalAngleMap, pieceAngleMap);
        orbitingIds = getIdsFromRasterPreProblem(curRasterNfp->getOrbitingName(), curRasterNfp->getOrbitingAngle(), pieceIndexMap, pieceTotalAngleMap, pieceAngleMap);
        if(staticIds.first == -1 || staticIds.second == -1 || orbitingIds.first == -1 || orbitingIds.second == -1) return false;

        // Create nofit polygon
        noFitPolygons->addRasterNoFitPolygon(staticIds.first, staticIds.second, orbitingIds.first, orbitingIds.second, curMountain);
    }

    // 5. Read problem scale
    this->scale = (*problem.crnfpbegin())->getScale();

    return true;
}
