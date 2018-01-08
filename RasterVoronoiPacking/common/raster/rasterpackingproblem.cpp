#include "rasterpackingproblem.h"
#include "../packingproblem.h"
#include <cmath>
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
    ids.first = 0; ids.second = 0;
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
	// 1. Load container information
	std::shared_ptr<RASTERPACKING::Container> problemContainer = *problem.cbegin();
	container = std::shared_ptr<RasterPackingItem>(new RasterPackingItem(-1, -1, 1, problemContainer->getPolygon()));
	container->setPieceName(problemContainer->getName());
	container->addAngleValue(0);
	qreal cMinX, cMaxX, cMinY, cMaxY; problemContainer->getPolygon()->getBoundingBox(cMinX, cMaxX, cMinY, cMaxY); container->setBoundingBox(cMinX, cMaxX, cMinY, cMaxY);
	totalArea = problem.getTotalItemsArea();

    // 2. Load container and items information
    int typeId = 0; int itemId = 0;
    for(QList<std::shared_ptr<RASTERPACKING::Piece>>::const_iterator it = problem.cpbegin(); it != problem.cpend(); it++, typeId++)
        for(uint mult = 0; mult < (*it)->getMultiplicity(); mult++, itemId++) {
            std::shared_ptr<RasterPackingItem> curItem = std::shared_ptr<RasterPackingItem>(new RasterPackingItem(itemId, typeId, (*it)->getOrientationsCount(), (*it)->getPolygon()));
            curItem->setPieceName((*it)->getName());
            for(QVector<unsigned int>::const_iterator it2 = (*it)->corbegin(); it2 != (*it)->corend(); it2++) curItem->addAngleValue(*it2);
			qreal minX, maxX, minY, maxY; (*it)->getPolygon()->getBoundingBox(minX, maxX, minY, maxY); curItem->setBoundingBox(minX, maxX, minY, maxY);
			items.append(curItem);
        }

    std::shared_ptr<RASTERPACKING::Container> container = *problem.ccbegin();
    std::shared_ptr<RASTERPACKING::Polygon> pol = container->getPolygon();
    qreal minX = (*pol->begin()).x();
	qreal minY = (*pol->begin()).y();
    qreal maxX = minX, maxY = minY;
    std::for_each(pol->begin(), pol->end(), [&minX, &maxX, &minY, &maxY](QPointF pt){
        if(pt.x() < minX) minX = pt.x();
        if(pt.x() > maxX) maxX = pt.x();
		if (pt.y() < minY) minY = pt.y();
		if (pt.y() > maxY) maxY = pt.y();
    });
    qreal rasterScale = (*problem.crnfpbegin())->getScale(); // FIXME: Global scale
    containerWidth = qRound(rasterScale*(maxX - minX));
	containerHeight = qRound(rasterScale*(maxY - minY));
    containerName = (*problem.ccbegin())->getName();

    // 3. Link the name and angle of the piece with the ids defined for the items
    QHash<QString, int> pieceIndexMap;
    QHash<int, int> pieceTotalAngleMap;
    QHash<QPair<int,int>, int> pieceAngleMap;
    mapPieceNameAngle(problem, pieceIndexMap, pieceTotalAngleMap, pieceAngleMap);

	// 4. Read Raster Data
	QVector<QPair<quint32, quint32>> sizes;
	QVector<QPoint> rps;
	quint32 *nfpData = loadBinaryNofitPolygons(problem.getNfpDataFileName(), sizes, rps);
	quint32 *curNfpData = nfpData;

    // 5. Load innerfit polygons
	innerFitPolygons = std::shared_ptr<RasterNoFitPolygonSet>(new RasterNoFitPolygonSet(items.length()));
    for(QList<std::shared_ptr<RASTERPACKING::RasterInnerFitPolygon>>::const_iterator it = problem.crifpbegin(); it != problem.crifpend(); it++) {
        std::shared_ptr<RASTERPACKING::RasterInnerFitPolygon> curRasterIfp = *it;
        // Create image. FIXME: Use data file instead?
        QImage curImage(curRasterIfp->getFileName());
		std::shared_ptr<RasterNoFitPolygon> curMap;
		if (nfpData == nullptr) curMap = std::shared_ptr<RasterNoFitPolygon>(new RasterNoFitPolygon(curImage, curRasterIfp->getReferencePoint()));
		else curMap = std::shared_ptr<RasterNoFitPolygon>(new RasterNoFitPolygon(curRasterIfp->getWidth(), curRasterIfp->getHeight(), curRasterIfp->getReferencePoint()));

        // Determine ids
        QPair<int,int> staticIds, orbitingIds;
        staticIds = getIdsFromRasterPreProblem(curRasterIfp->getStaticName(), curRasterIfp->getStaticAngle(), pieceIndexMap, pieceTotalAngleMap, pieceAngleMap);
        orbitingIds = getIdsFromRasterPreProblem(curRasterIfp->getOrbitingName(), curRasterIfp->getOrbitingAngle(), pieceIndexMap, pieceTotalAngleMap, pieceAngleMap);

        // Create nofit polygon
        innerFitPolygons->addRasterNoFitPolygon(staticIds.first, staticIds.second, orbitingIds.first, orbitingIds.second, curMap);
    }

    // 6. Load nofit polygons
	noFitPolygons = std::shared_ptr<RasterNoFitPolygonSet>(new RasterNoFitPolygonSet(items.length()));
	int i = 0;
	for(QVector<std::shared_ptr<RASTERPACKING::RasterNoFitPolygon>>::const_iterator it = problem.crnfpbegin(); it != problem.crnfpend(); it++, i++) {
        std::shared_ptr<RASTERPACKING::RasterNoFitPolygon> curRasterNfp = *it;
        // Create image. FIXME: Use data file instead?
        QImage curImage(curRasterNfp->getFileName());
		std::shared_ptr<RasterNoFitPolygon> curMountain;
		if (nfpData == nullptr) curMountain = std::shared_ptr<RasterNoFitPolygon>(new RasterNoFitPolygon(curImage, curRasterNfp->getReferencePoint()));
		else {
			curMountain = std::shared_ptr<RasterNoFitPolygon>(new RasterNoFitPolygon(curNfpData, sizes[i].first, sizes[i].second, rps[i]));
			curNfpData += sizes[i].first * sizes[i].second;
		}

        // Determine ids
        QPair<int,int> staticIds, orbitingIds;
        staticIds = getIdsFromRasterPreProblem(curRasterNfp->getStaticName(), curRasterNfp->getStaticAngle(), pieceIndexMap, pieceTotalAngleMap, pieceAngleMap);
        orbitingIds = getIdsFromRasterPreProblem(curRasterNfp->getOrbitingName(), curRasterNfp->getOrbitingAngle(), pieceIndexMap, pieceTotalAngleMap, pieceAngleMap);
        if(staticIds.first == -1 || staticIds.second == -1 || orbitingIds.first == -1 || orbitingIds.second == -1) return false;

        // Create nofit polygon
        noFitPolygons->addRasterNoFitPolygon(staticIds.first, staticIds.second, orbitingIds.first, orbitingIds.second, curMountain);
    }

    // 7. Read problem scale
    this->scale = (*problem.crnfpbegin())->getScale();

	// 8. Read max size for container
	qreal scaledMaxWidth = problem.getMaxLength() * this->scale;
	qreal scaledMaxHeight = problem.getMaxWidth() * this->scale;
	if (std::floor(scaledMaxWidth) == scaledMaxWidth) this->maxWidth = qRound(scaledMaxWidth);
	else this->maxWidth = std::ceil(scaledMaxWidth) + 1; // FIXME: Determine if +1 is necessary
	if (std::floor(scaledMaxHeight) == scaledMaxHeight) this->maxHeight = qRound(scaledMaxHeight);
	else this->maxHeight = std::ceil(scaledMaxHeight) + 1; // FIXME: Determine if +1 is necessary

    return true;
}

quint32 *RasterPackingProblem::loadBinaryNofitPolygons(QString fileName, QVector<QPair<quint32, quint32>> &sizes, QVector<QPoint> &rps) {
	QFile file(fileName);
	if(!file.open(QIODevice::ReadOnly)) return nullptr;
	QDataStream in(&file);
	in.setByteOrder(QDataStream::LittleEndian);
	quint32 numNfps;
	in >> numNfps;
	sizes.reserve(numNfps); rps.reserve(numNfps);
	quint32 numElements = 0;
	for (unsigned int i = 0; i < numNfps; i++) {
		quint32 width, height, rpx, rpy;
		in >> width >> height >> rpx >> rpy;
		numElements += width * height;
		sizes << QPair<quint32, quint32>(width, height);
		rps << QPoint(rpx, rpy);
	}
	quint32 *data = new quint32[numElements];
	quint64 numBytes = numElements * sizeof(quint32);
	quint64 maxInt = (quint32)std::numeric_limits<int>::max();
	if (numBytes > maxInt) {
		char *dataPtr = (char *)data;
		for (; numBytes > maxInt; numBytes -= maxInt) {
			in.readRawData(dataPtr, maxInt);
			dataPtr += maxInt;
		}
		in.readRawData(dataPtr, numBytes);
	}
	else in.readRawData((char *)data, numBytes);
	file.close();
	return data;
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
				QPointF newOffset = QTransform().rotate(clusterAngle).map(item.offset);

				// Get new angle
				int newAngle = (this->originalProblem->getItem(item.id)->getAngleValue(item.angle) + clusterAngle) % 360;
				// Find corrresponding orientation
				int newOrientation = this->originalProblem->getItem(item.id)->getOrientationFromAngle(newAngle);
				if (newOrientation == -1) {
					// Find mirror rotation. FIXME: Only works with increments of 90 degrees
					if (this->originalProblem->getItem(item.id)->getAngleCount() == 1) newOrientation = 0;
					else {
						for (unsigned int i = 0; i < this->originalProblem->getItem(item.id)->getAngleCount(); i++) {
							if ((qAbs(newAngle - 180) % 360 == this->originalProblem->getItem(item.id)->getAngleValue(i))
								|| ((newAngle + 180) % 360 == this->originalProblem->getItem(item.id)->getAngleValue(i))) {
								newOrientation = i;
								break;
							}
						}
					}
					Q_ASSERT(newOrientation != -1);
					newAngle = this->originalProblem->getItem(item.id)->getAngleValue(newOrientation);

					// Get item bounding box
					qreal minX, minY, maxX, maxY;
					this->originalProblem->getItem(item.id)->getBoundingBox(minX, maxX, minY, maxY);

					// Determine offset of the reference point
					int deltaAngle = (this->originalProblem->getItem(item.id)->getAngleValue(item.angle) + clusterAngle) - newAngle;
					QPointF center = QPointF(maxX - minX, maxY - minY) / 2 + QPointF(minX, minY);
					QPointF referencePointsOffset = -QTransform().translate(center.x(), center.y()).rotate(deltaAngle).map(-center);

					// Adjust offset to match mirror posvisuition
					newOffset = newOffset + QTransform().scale(this->originalProblem->getScale(), this->originalProblem->getScale()).rotate(newAngle).map(referencePointsOffset);
				}

				// Set position and orientation
				QPoint newOffSetGrid = QPoint(qRound(newOffset.x()), qRound(newOffset.y()));
				solution.setPosition(item.id, oldSolution.getPosition(it.key()) + newOffSetGrid);
				solution.setOrientation(item.id, newOrientation);
			}
			continue;
		}
		solution.setPosition(currCluster.first().id, oldSolution.getPosition(it.key()));
		solution.setOrientation(currCluster.first().id, oldSolution.getOrientation(it.key()));
	}
}


quint32 RasterPackingProblem::getDistanceValue(int itemId1, QPoint pos1, int orientation1, int itemId2, QPoint pos2, int orientation2) {
	return getNfpValue(itemId1, pos1, orientation1, itemId2, pos2, orientation2);
}

bool RasterPackingProblem::areOverlapping(int itemId1, QPoint pos1, int orientation1, int itemId2, QPoint pos2, int orientation2) {
	return getNfpValue(itemId1, pos1, orientation1, itemId2, pos2, orientation2) != 0;
}

quint32 RasterPackingProblem::getNfpValue(int itemId1, QPoint pos1, int orientation1, int itemId2, QPoint pos2, int orientation2) {
	std::shared_ptr<RasterNoFitPolygon> curNfp = noFitPolygons->getRasterNoFitPolygon(getItemType(itemId1), orientation1, getItemType(itemId2), orientation2);
	QPoint relPos = pos2 - pos1 + curNfp->getOrigin();

	if (relPos.x() < 0 || relPos.x() > curNfp->width() - 1 || relPos.y() < 0 || relPos.y() > curNfp->height() - 1) return 0;
	return curNfp->getPixel(relPos.x(), relPos.y());
}

// TODO: Use QRect
void RasterPackingProblem::getIfpBoundingBox(int itemId, int orientation, int &bottomLeftX, int &bottomLeftY, int &topRightX, int &topRightY) {
	std::shared_ptr<RasterNoFitPolygon> ifp = innerFitPolygons->getRasterNoFitPolygon(0, 0, getItemType(itemId), orientation);
	bottomLeftX = -ifp->getOriginX();
	bottomLeftY = -ifp->getOriginY();
	topRightX = bottomLeftX + ifp->width() - 1;
	topRightY = bottomLeftY + ifp->height() - 1;
}

qreal RasterPackingProblem::getItemsMaxX(RasterPackingSolution &solution) {
	QVector<qreal> solutionIndividualMaxXs;
	std::transform(items.begin(), items.end(), std::back_inserter(solutionIndividualMaxXs),
		[&solution, this](std::shared_ptr<RasterPackingItem> const &hs) -> qreal { return solution.getPosition(hs->getId()).x() / scale + hs->getMaxX(solution.getOrientation(hs->getId())); });
	return *std::max_element(solutionIndividualMaxXs.begin(), solutionIndividualMaxXs.end());
}

qreal RasterPackingProblem::getItemsMinX(RasterPackingSolution &solution) {
	QVector<qreal> solutionIndividualMinXs;
	std::transform(items.begin(), items.end(), std::back_inserter(solutionIndividualMinXs),
		[&solution, this](std::shared_ptr<RasterPackingItem> const &hs) -> qreal { return solution.getPosition(hs->getId()).x() / scale + hs->getMinX(solution.getOrientation(hs->getId())); });
	return *std::min_element(solutionIndividualMinXs.begin(), solutionIndividualMinXs.end());
}

qreal RasterPackingProblem::getItemsMaxY(RasterPackingSolution &solution) {
	QVector<qreal> solutionIndividualMaxYs;
	std::transform(items.begin(), items.end(), std::back_inserter(solutionIndividualMaxYs),
		[&solution, this](std::shared_ptr<RasterPackingItem> const &hs) -> qreal { return solution.getPosition(hs->getId()).y() / scale + hs->getMaxY(solution.getOrientation(hs->getId())); });
	return *std::max_element(solutionIndividualMaxYs.begin(), solutionIndividualMaxYs.end());
}

qreal RasterPackingProblem::getItemsMinY(RasterPackingSolution &solution) {
	QVector<qreal> solutionIndividualMinYs;
	std::transform(items.begin(), items.end(), std::back_inserter(solutionIndividualMinYs),
		[&solution, this](std::shared_ptr<RasterPackingItem> const &hs) -> qreal { return solution.getPosition(hs->getId()).y() / scale + hs->getMinY(solution.getOrientation(hs->getId())); });
	return *std::min_element(solutionIndividualMinYs.begin(), solutionIndividualMinYs.end());
}

qreal RasterPackingProblem::getOriginalWidth() {
	return container->getMaxX(0) - container->getMinX(0);
}
qreal RasterPackingProblem::getOriginalHeight() {
	return container->getMaxY(0) - container->getMinY(0);
}

qreal RasterPackingProblem::getCurrentWidth(RasterPackingSolution &solution) {
	return getItemsMaxX(solution) - getItemsMinX(solution);
}

qreal RasterPackingProblem::getCurrentHeight(RasterPackingSolution &solution) {
	return getItemsMaxY(solution) - getItemsMinY(solution);
}

qreal RasterPackingProblem::getDensity(RasterPackingSolution &solution) {
	qreal curWidth = getItemsMaxX(solution) - getItemsMinX(solution);
	qreal originalHeight = container->getMaxY(0) - container->getMinY(0);
	return this->totalArea / (curWidth * originalHeight);
}

qreal RasterPackingProblem::getSquareDensity(RasterPackingSolution &solution) {
	qreal curWidth = getItemsMaxX(solution) - getItemsMinX(solution);
	qreal curHeight = getItemsMaxY(solution) - getItemsMinY(solution);
	qreal curDim = std::max(curWidth, curHeight);
	return this->totalArea / (curDim * curDim);
}

qreal RasterPackingProblem::getRectangularDensity(RasterPackingSolution &solution) {
	qreal curWidth = getItemsMaxX(solution) - getItemsMinX(solution);
	qreal curHeight = getItemsMaxY(solution) - getItemsMinY(solution);
	return this->totalArea / (curWidth * curHeight);
}