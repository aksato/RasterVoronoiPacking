#include "rasterpackingproblem.h"
#include "../packingproblem.h"
#include <QDir>
#include <QFileInfo>

#define NUM_ORIENTATIONS 4 // FIXME: Get form problem

using namespace RASTERVORONOIPACKING;

RasterPackingProblem::RasterPackingProblem()
{
}

RasterPackingProblem::RasterPackingProblem(RASTERPACKING::PackingProblem &problem) {
    this->load(problem);
}

void mapPieceNameAngle(RASTERPACKING::PackingProblem &problem, QHash<QString, int> &pieceIndexMap, QHash<int, int> &pieceTotalAngleMap, QHash<QPair<int,int>, int> &pieceAngleMap) {
    int id = 0;
    for(QList<std::shared_ptr<RASTERPACKING::Piece>>::const_iterator it = problem.cpbegin(); it != problem.cpend(); it++) {
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

int *getMatrixFromQImage(QImage image) {
	int *ans = (int*)malloc(image.width()*image.height()*sizeof(int));
	for (int j = 0; j < image.height(); j++)
	for (int i = 0; i < image.width(); i++)
		ans[j*image.width() + i] = image.pixelIndex(i, j);
	return ans;
}

bool RasterPackingProblem::load(RASTERPACKING::PackingProblem &problem) {
    // 1. Load items information
    int typeId = 0; int itemId = 0;
    for(QList<std::shared_ptr<RASTERPACKING::Piece>>::const_iterator it = problem.cpbegin(); it != problem.cpend(); it++, typeId++)
        for(uint mult = 0; mult < (*it)->getMultiplicity(); mult++, itemId++) {
            std::shared_ptr<RasterPackingItem> curItem = std::shared_ptr<RasterPackingItem>(new RasterPackingItem(itemId, typeId, (*it)->getOrientationsCount()));
            curItem->setPieceName((*it)->getName());
            for(QVector<unsigned int>::const_iterator it2 = (*it)->corbegin(); it2 != (*it)->corend(); it2++) curItem->addAngleValue(*it2);
			int minX, maxX, minY, maxY; (*it)->getPolygon()->getBoundingBox(minX, maxX, minY, maxY); curItem->setBoundingBox(minX, maxX, minY, maxY);
			items.append(curItem);
        }

    std::shared_ptr<RASTERPACKING::Container> container = *problem.ccbegin();
    std::shared_ptr<RASTERPACKING::Polygon> pol = container->getPolygon();
    qreal minX = (*pol->begin()).x();
    qreal maxX = minX;
    std::for_each(pol->begin(), pol->end(), [&minX, &maxX](QPointF pt){
        if(pt.x() < minX) minX = pt.x();
        if(pt.x() > maxX) maxX = pt.x();
    });
    qreal rasterScale = (*problem.crnfpbegin())->getScale(); // FIXME: Global scale
    containerWidth = qRound(rasterScale*(maxX - minX));
    containerName = (*problem.ccbegin())->getName();

    // 2. Link the name and angle of the piece with the ids defined for the items
    QHash<QString, int> pieceIndexMap;
    QHash<int, int> pieceTotalAngleMap;
    QHash<QPair<int,int>, int> pieceAngleMap;
    mapPieceNameAngle(problem, pieceIndexMap, pieceTotalAngleMap, pieceAngleMap);

    // 3. Load innerfit polygons
    innerFitPolygons = std::shared_ptr<RasterNoFitPolygonSet>(new RasterNoFitPolygonSet);
    for(QList<std::shared_ptr<RASTERPACKING::RasterInnerFitPolygon>>::const_iterator it = problem.crifpbegin(); it != problem.crifpend(); it++) {
        std::shared_ptr<RASTERPACKING::RasterInnerFitPolygon> curRasterIfp = *it;
        // Create image. FIXME: Use data file instead?
        QImage curImage(curRasterIfp->getFileName());
        std::shared_ptr<RasterNoFitPolygon> curMap(new RasterNoFitPolygon(curImage, curRasterIfp->getReferencePoint(), 1.0));

        // Determine ids
        QPair<int,int> staticIds, orbitingIds;
        staticIds = getIdsFromRasterPreProblem(curRasterIfp->getStaticName(), curRasterIfp->getStaticAngle(), pieceIndexMap, pieceTotalAngleMap, pieceAngleMap);
        orbitingIds = getIdsFromRasterPreProblem(curRasterIfp->getOrbitingName(), curRasterIfp->getOrbitingAngle(), pieceIndexMap, pieceTotalAngleMap, pieceAngleMap);

        // Create nofit polygon
        innerFitPolygons->addRasterNoFitPolygon(staticIds.first, staticIds.second, orbitingIds.first, orbitingIds.second, curMap);
    }

    // 4. Load nofit polygons
    noFitPolygons = std::shared_ptr<RasterNoFitPolygonSet>(new RasterNoFitPolygonSet);
    for(QList<std::shared_ptr<RASTERPACKING::RasterNoFitPolygon>>::const_iterator it = problem.crnfpbegin(); it != problem.crnfpend(); it++) {
        std::shared_ptr<RASTERPACKING::RasterNoFitPolygon> curRasterNfp = *it;
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

		// Alloc in GPU
		int static1DId, orbitingDId;
		static1DId = staticIds.first * NUM_ORIENTATIONS + staticIds.second; orbitingDId = orbitingIds.first*NUM_ORIENTATIONS + orbitingIds.second;  // FIXME: get number of orientations
    }

    // 5. Read problem scale
    this->scale = (*problem.crnfpbegin())->getScale();

    return true;
}

void RasterPackingProblem::getProblemGPUMemRequirements(RASTERPACKING::PackingProblem &problem, size_t &ifpTotalMem, size_t &ifpMaxMem, size_t &nfpTotalMem) {
	unsigned int ifpCount = 0; ifpTotalMem = 0;  ifpMaxMem = 0;
	for (QList<std::shared_ptr<RASTERPACKING::RasterInnerFitPolygon>>::const_iterator it = problem.crifpbegin(); it != problem.crifpend(); it++) {
		std::shared_ptr<RASTERPACKING::RasterInnerFitPolygon> curRasterIfp = *it;
		// Create image. FIXME: Use data file instead?
		QImage curImage(curRasterIfp->getFileName());

		// Determine memory space
		size_t curIfpMemSize = curImage.width()*curImage.height()*sizeof(qreal);
		if (ifpMaxMem == 0 || curIfpMemSize > ifpMaxMem) ifpMaxMem = curIfpMemSize; 
		ifpTotalMem += curIfpMemSize; ifpCount++;
	}

	unsigned int nfpCount = 0; nfpTotalMem = 0;
	for (QList<std::shared_ptr<RASTERPACKING::RasterNoFitPolygon>>::const_iterator it = problem.crnfpbegin(); it != problem.crnfpend(); it++) {
		std::shared_ptr<RASTERPACKING::RasterNoFitPolygon> curRasterNfp = *it;
		// Create image. FIXME: Use data file instead?
		QImage curImage(curRasterNfp->getFileName());

		// Determine memory space
		nfpTotalMem += curImage.width()*curImage.height()*sizeof(int); nfpCount++;
	}
}

bool RasterPackingClusterProblem::load(RASTERPACKING::PackingProblem &problem) {
	if (!RasterPackingProblem::load(problem)) return false;

	// Load original problem
	RASTERPACKING::PackingProblem nonClusteredProblem;
	QDir::setCurrent(QFileInfo(problem.getOriginalProblem()).absolutePath());
	if (!nonClusteredProblem.load(problem.getOriginalProblem())) return false;
	this->originalProblem = std::shared_ptr<RASTERVORONOIPACKING::RasterPackingProblem>(new RASTERVORONOIPACKING::RasterPackingProblem);
	if(!this->originalProblem->load(nonClusteredProblem)) return false;

	// Create map of original item indexes -> piece names
	QMap<int, QString> originalIdToNameMap;
	for (QVector<std::shared_ptr<RasterPackingItem>>::iterator it = this->originalProblem->ibegin(); it != this->originalProblem->iend(); it++) {
		std::shared_ptr<RasterPackingItem> item = *it;
		originalIdToNameMap.insert(item->getId(), item->getPieceName());
	}
	
	// For each item of the clustered problem, determine its respective cluster
	QMap<int, RasterPackingCluster> clustersMaptest;
	foreach(std::shared_ptr<RasterPackingItem> item, items) {
		int originalPieceId = item->getId();
		RASTERPACKING::CLUSTERING::Cluster currCluster = problem.getCluster(item->getPieceName());
		
		RasterPackingCluster translatedCluster;
		foreach(RASTERPACKING::CLUSTERING::ClusterPiece clPiece, currCluster) {
			// Get piece id and remove from map
			int pieceId = originalIdToNameMap.key(clPiece.pieceName);
			originalIdToNameMap.remove(pieceId);

			// Get angle
			unsigned int angleId = 0;
			for (; angleId < this->originalProblem->getItem(pieceId)->getAngleCount(); angleId++)
			if (this->originalProblem->getItem(pieceId)->getAngleValue(angleId) == clPiece.angle)
				break;

			// Create cluster item object
			RasterPackingClusterItem rasterClusterItem(clPiece.pieceName, pieceId, angleId, QPoint(qRound(clPiece.offset.x()*scale), qRound(clPiece.offset.y()*scale)));
			translatedCluster.push_back(rasterClusterItem);
		}
		clustersMap.insert(originalPieceId, translatedCluster);
	}
	return true;
}

void RasterPackingClusterProblem::convertSolution(RASTERVORONOIPACKING::RasterPackingSolution &solution) {
	RASTERVORONOIPACKING::RasterPackingSolution oldSolution = solution;
	solution.reset(this->originalProblem->count());

	// Recreate solution
	for (QMap<int, RasterPackingCluster>::iterator it = clustersMap.begin(); it != clustersMap.end(); it++) {
		RASTERVORONOIPACKING::RasterPackingCluster currCluster = *it;
		if (currCluster.length() != 1) {
			foreach(RASTERVORONOIPACKING::RasterPackingClusterItem item, currCluster) {
				// Rotate offset
				int clusterAngle = getItem(it.key())->getAngleValue(oldSolution.getOrientation(it.key()));
				QPoint newOffset = QTransform().rotate(clusterAngle).map(item.offset);

				// Get new angle
				int newAngle = (this->originalProblem->getItem(item.id)->getAngleValue(item.angle) + clusterAngle) % 360;
				// Find corrresponding orientation
				int newOrientation = this->originalProblem->getItem(item.id)->getOrientationFromAngle(newAngle);
				if (newOrientation == -1) {
					// Find mirror rotation
					for (int i = 0; i < this->originalProblem->getItem(item.id)->getAngleCount(); i++) {
						if ((qAbs(newAngle - 180) % 360 == this->originalProblem->getItem(item.id)->getAngleValue(i))
							|| ((newAngle + 180) % 360 == this->originalProblem->getItem(item.id)->getAngleValue(i))) {
							newOrientation = i;
							newAngle = this->originalProblem->getItem(item.id)->getAngleValue(i);
							break;
						}
					}
					Q_ASSERT(newOrientation != -1);
					// Get item bounding box
					int minX, minY, maxX, maxY;
					this->originalProblem->getItem(item.id)->getBoundingBox(minX, maxX, minY, maxY);
					// Adjust offset to match mirror position
					newOffset = newOffset - QTransform().scale(this->originalProblem->getScale(), this->originalProblem->getScale()).rotate(newAngle).map(QPoint(maxX + minX, maxY + minY));
				}

				// Set position and orientation
				solution.setPosition(item.id, oldSolution.getPosition(it.key()) + newOffset);
				solution.setOrientation(item.id, newOrientation);
			}
			continue;
		}
		solution.setPosition(currCluster.first().id, oldSolution.getPosition(it.key()));
		solution.setOrientation(currCluster.first().id, oldSolution.getOrientation(it.key()));
	}
}