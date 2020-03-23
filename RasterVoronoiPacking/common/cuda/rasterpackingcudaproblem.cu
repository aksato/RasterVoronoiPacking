#include "rasterpackingcudaproblem.h"
#include "../packingproblem.h"
#include <cmath>
#include <QDir>
#include <QFileInfo>
#include <iostream>
#define NUM_ORIENTATIONS 4 // FIXME: Get form problem

using namespace RASTERVORONOIPACKING;

RasterPackingCudaProblem::RasterPackingCudaProblem()
{
}

RasterPackingCudaProblem::RasterPackingCudaProblem(RASTERPACKING::PackingProblem &problem) {
	this->load(problem);
}


bool RasterPackingCudaProblem::load(RASTERPACKING::PackingProblem &problem) {
	// 1. Load container information
	std::shared_ptr<RASTERPACKING::Container> problemContainer = *problem.cbegin();
	container = std::shared_ptr<RasterPackingItem>(new RasterPackingItem(-1, -1, 1, problemContainer->getPolygon()));
	container->setPieceName(problemContainer->getName());
	container->addAngleValue(0);
	qreal cMinX, cMaxX, cMinY, cMaxY; problemContainer->getPolygon()->getBoundingBox(cMinX, cMaxX, cMinY, cMaxY); container->setBoundingBox(cMinX, cMaxX, cMinY, cMaxY);
	totalArea = problem.getTotalItemsArea();

	// 2. Load container and items information
	int typeId = 0; int itemId = 0;
	for (QList<std::shared_ptr<RASTERPACKING::Piece>>::const_iterator it = problem.cpbegin(); it != problem.cpend(); it++, typeId++) {
		multiplicity.push_back((*it)->getMultiplicity());
		for (uint mult = 0; mult < (*it)->getMultiplicity(); mult++, itemId++) {
			std::shared_ptr<RasterPackingItem> curItem = std::shared_ptr<RasterPackingItem>(new RasterPackingItem(itemId, typeId, (*it)->getOrientationsCount(), (*it)->getPolygon()));
			curItem->setPieceName((*it)->getName());
			for (QVector<unsigned int>::const_iterator it2 = (*it)->corbegin(); it2 != (*it)->corend(); it2++) curItem->addAngleValue(*it2);
			qreal minX, maxX, minY, maxY; (*it)->getPolygon()->getBoundingBox(minX, maxX, minY, maxY); curItem->setBoundingBox(minX, maxX, minY, maxY);
			items.append(curItem);
		}
	}
	itemTypeCount = std::distance(problem.cpbegin(), problem.cpend());

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
	quint32 *nfpData = loadBinaryNofitPolygonsOnDevice(problem, sizes, rps);
	quint32 *curNfpData = nfpData;

    // 5. Load innerfit polygons
	innerFitPolygons = std::shared_ptr<RasterNoFitPolygonSet>(new RasterNoFitPolygonSet(items.length()));
    for(QList<std::shared_ptr<RASTERPACKING::RasterInnerFitPolygon>>::const_iterator it = problem.crifpbegin(); it != problem.crifpend(); it++) {
        std::shared_ptr<RASTERPACKING::RasterInnerFitPolygon> curRasterIfp = *it;
		std::shared_ptr<RasterNoFitPolygon> curMap;
		curMap = std::shared_ptr<RasterNoFitPolygon>(new RasterNoFitPolygon(curRasterIfp->getWidth(), curRasterIfp->getHeight(), curRasterIfp->getReferencePoint()));

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
		std::shared_ptr<RasterNoFitPolygon> curMountain;

        // Determine ids
        QPair<int,int> staticIds, orbitingIds;
        staticIds = getIdsFromRasterPreProblem(curRasterNfp->getStaticName(), curRasterNfp->getStaticAngle(), pieceIndexMap, pieceTotalAngleMap, pieceAngleMap);
        orbitingIds = getIdsFromRasterPreProblem(curRasterNfp->getOrbitingName(), curRasterNfp->getOrbitingAngle(), pieceIndexMap, pieceTotalAngleMap, pieceAngleMap);
        if(staticIds.first == -1 || staticIds.second == -1 || orbitingIds.first == -1 || orbitingIds.second == -1) return false;

        // Create nofit polygon
		curMountain = std::shared_ptr<RasterNoFitPolygon>(new RasterNoFitPolygon(curNfpData, sizes[i].first, sizes[i].second, rps[i]));

		if (problem.getDataSymmetry() == RASTERPACKING::Symmetry::NONE) {
			noFitPolygons->addRasterNoFitPolygon(staticIds.first, staticIds.second, orbitingIds.first, orbitingIds.second, curMountain);
			curNfpData += sizes[i].first * sizes[i].second;
		}
		else if (problem.getDataSymmetry() == RASTERPACKING::Symmetry::PAIR) {
			if (staticIds.first <= orbitingIds.first) {
				// Add original nfp
				noFitPolygons->addRasterNoFitPolygon(staticIds.first, staticIds.second, orbitingIds.first, orbitingIds.second, curMountain);
				// Add symmetry
				if (staticIds.first != orbitingIds.first) {
					std::shared_ptr<RasterNoFitPolygon> curMountainFlipped = std::shared_ptr<RasterNoFitPolygon>(new RasterNoFitPolygonFlipped(curNfpData, sizes[i].first, sizes[i].second, rps[i]));
					noFitPolygons->addRasterNoFitPolygon(orbitingIds.first, orbitingIds.second, staticIds.first, staticIds.second, curMountainFlipped);
				}
			}
			curNfpData += sizes[i].first * sizes[i].second;
		}
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

quint32 *RasterPackingCudaProblem::loadBinaryNofitPolygonsOnDevice(RASTERPACKING::PackingProblem& problem, QVector<QPair<quint32, quint32>> &sizes, QVector<QPoint> &rps) {
	quint32* data = problem.loadBinaryNofitPolygons(sizes, rps);
	quint32 numElements = 0;
	for (auto s : sizes) numElements += s.first * s.second;

	//
	quint32 *d_data;
	quint64 numBytes = numElements * sizeof(quint32);
	cudaMalloc((void **)&d_data, numBytes);
	cudaDeviceSynchronize();

	auto error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("CUDA error loading %.2f MB binary NFPs on device: %s\n", numBytes / (1024.0 * 1024.0), cudaGetErrorString(error));
		// show memory usage of GPU
		size_t free_byte, total_byte;
		auto cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
		if (cudaSuccess != cuda_status) printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
		else {
			double free_db = (double)free_byte;
			double total_db = (double)total_byte;
			double used_db = total_db - free_db;
			printf("Memory report:: used = %.2f MB, free = %.2f MB, total = %.2f MB\n", used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
		}
	}
	else cudaMemcpy(d_data, data, numBytes, cudaMemcpyHostToDevice);

	return d_data;
}