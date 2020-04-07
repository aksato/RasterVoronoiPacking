#include "rasteroverlapevaluatorfull.h"

using namespace RASTERVORONOIPACKING;

RasterTotalOverlapMapEvaluatorFull::RasterTotalOverlapMapEvaluatorFull(std::shared_ptr<RasterPackingProblem> _problem, bool cuttingStock) : RasterTotalOverlapMapEvaluatorGLS(_problem, false, cuttingStock) {}

RasterTotalOverlapMapEvaluatorFull::RasterTotalOverlapMapEvaluatorFull(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<GlsWeightSet> _glsWeights, bool cuttingStock) : RasterTotalOverlapMapEvaluatorGLS(_problem, _glsWeights, false, cuttingStock) {}

// Determines the item total overlap map for a given orientation in a solution
std::shared_ptr<TotalOverlapMap> RasterTotalOverlapMapEvaluatorFull::getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution) {
	std::shared_ptr<TotalOverlapMap> currrentPieceMap = maps.getOverlapMap(itemId, orientation);
	currrentPieceMap->reset();
	std::shared_ptr<ItemRasterNoFitPolygonSet> curItemNfpSet = problem->getNfps()->getItemRasterNoFitPolygonSet(problem->getItemType(itemId), orientation);
	quint32 *mapPointer = currrentPieceMap->getData();
	for(int i = 0; i < currrentPieceMap->getWidth(); i++)
		for (int j = 0; j < currrentPieceMap->getHeight(); j++) {
			for (int k = 0; k < problem->count(); k++) {
				if (k == itemId) continue;
				std::shared_ptr<RasterNoFitPolygon> nfp = problem->getNfps()->getRasterNoFitPolygon(problem->getItemType(k), solution.getOrientation(k), problem->getItemType(itemId), orientation);
				QPoint relativeOrigin = currrentPieceMap->getReferencePoint() + solution.getPosition(k) - nfp->getFlipMultiplier()*nfp->getOrigin() + QPoint(nfp->width() - 1, nfp->height() - 1)*(nfp->getFlipMultiplier() - 1) / 2;
				if (i < relativeOrigin.x() || i > relativeOrigin.x() + nfp->width() - 1 || j < relativeOrigin.y() || j > relativeOrigin.y() + nfp->height() - 1) continue;
				else {
					quint32* nfpPointer = nfp->getPixelRef(i - relativeOrigin.x(), j - relativeOrigin.y());
					*mapPointer += *nfpPointer * getWeight(itemId, k);
				}
			}
			mapPointer++;
		}
	return currrentPieceMap;
}

std::shared_ptr<TotalOverlapMap> RasterTotalOverlapMapEvaluatorFull::getPartialTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution, QList<int> &placedItems)  {
	// TODO
	std::shared_ptr<TotalOverlapMap> currrentPieceMap = maps.getOverlapMap(itemId, orientation);
	return currrentPieceMap;
}