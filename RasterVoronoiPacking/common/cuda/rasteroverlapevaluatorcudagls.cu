#include "cuda/rasteroverlapevaluatorcudagls.h"

using namespace RASTERVORONOIPACKING;

RasterTotalOverlapMapEvaluatorCudaGLS::RasterTotalOverlapMapEvaluatorCudaGLS(std::shared_ptr<RasterPackingProblem> _problem) : RasterTotalOverlapMapEvaluator(_problem, false), cudamaps(_problem->count()) {
	populateMaps();
}

RasterTotalOverlapMapEvaluatorCudaGLS::RasterTotalOverlapMapEvaluatorCudaGLS(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<GlsWeightSet> _glsWeights) : RasterTotalOverlapMapEvaluator(_problem, _glsWeights, false), cudamaps(_problem->count()) {
	populateMaps();
}

void RasterTotalOverlapMapEvaluatorCudaGLS::populateMaps() {
	for (int itemId = 0; itemId < problem->count(); itemId++) {
		for (uint angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
			std::shared_ptr<TotalOverlapMapCuda> curMap = std::shared_ptr<TotalOverlapMapCuda>(new TotalOverlapMapCuda(problem->getIfps()->getRasterNoFitPolygon(0, 0, problem->getItemType(itemId), angle), -1));
			cudamaps.addOverlapMap(itemId, angle, curMap);
			// FIXME: Delete innerift polygons as they are used to release memomry
		}
	}
}

void RasterTotalOverlapMapEvaluatorCudaGLS::updateMapsLength(int pixelWidth) {
	int deltaPixel = problem->getContainerWidth() - pixelWidth;
	cudamaps.setShrinkVal(deltaPixel);
	for (int itemId = 0; itemId < problem->count(); itemId++)
		for (uint angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
			std::shared_ptr<TotalOverlapMap> curMap = cudamaps.getOverlapMap(itemId, angle);
			curMap->setRelativeWidth(deltaPixel);
		}
}

std::shared_ptr<TotalOverlapMap> RasterTotalOverlapMapEvaluatorCudaGLS::getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution) {
	std::shared_ptr<TotalOverlapMap> currrentPieceMap = cudamaps.getOverlapMap(itemId, orientation);
	currrentPieceMap->reset();
	std::shared_ptr<ItemRasterNoFitPolygonSet> curItemNfpSet = problem->getNfps()->getItemRasterNoFitPolygonSet(problem->getItemType(itemId), orientation);
	for (int i = 0; i < problem->count(); i++) {
		if (i == itemId) continue;
		currrentPieceMap->addVoronoi(i, curItemNfpSet->getRasterNoFitPolygon(problem->getItemType(i), solution.getOrientation(i)), solution.getPosition(i), getWeight(itemId, i));
	}
	return currrentPieceMap;
	//std::shared_ptr<TotalOverlapMap> dummyPieceMap = std::shared_ptr<TotalOverlapMap>(new TotalOverlapMap(currrentPieceMap->getRect(), currrentPieceMap->getCuttingStockLength()));
	//cudaMemcpy(dummyPieceMap->getData(), currrentPieceMap->getData(), currrentPieceMap->getHeight() * currrentPieceMap->getWidth() * sizeof(quint32), cudaMemcpyDeviceToHost);
	//return dummyPieceMap;
}