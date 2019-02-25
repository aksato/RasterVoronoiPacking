#include "rasteroverlapevaluatormatrixgls.h"
#include <Eigen/Core>

using namespace RASTERVORONOIPACKING;

void RasterTotalOverlapMapEvaluatorMatrixGLS::createWeigthVector(int itemId, Eigen::Matrix< unsigned int, Eigen::Dynamic, 1 > &vec) {
	for (int i = 0; i < problem->count(); i++) {
		if (i == itemId) continue;
		vec.coeffRef(i) = glsWeights->getWeight(itemId, i);
	}
}

std::shared_ptr<TotalOverlapMap> RasterTotalOverlapMapEvaluatorMatrixGLS::getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution) {
	std::shared_ptr<TotalOverlapMap> currrentPieceMap = maps.getOverlapMap(itemId, orientation);
	Eigen::Matrix< unsigned int, Eigen::Dynamic, Eigen::Dynamic > overlapMatrix = Eigen::Matrix< unsigned int, Eigen::Dynamic, Eigen::Dynamic >::Zero(currrentPieceMap->getHeight() * currrentPieceMap->getWidth(), problem->count());

	std::shared_ptr<ItemRasterNoFitPolygonSet> curItemNfpSet = problem->getNfps()->getItemRasterNoFitPolygonSet(problem->getItemType(itemId), orientation);
	for (int i = 0; i < problem->count(); i++) {
		if (i == itemId) continue;
		currrentPieceMap->addToMatrix(i, curItemNfpSet->getRasterNoFitPolygon(problem->getItemType(i), solution.getOrientation(i)), solution.getPosition(i), overlapMatrix);
	}

	Eigen::Matrix< unsigned int, Eigen::Dynamic, 1 >  weightVec = Eigen::Matrix< unsigned int, Eigen::Dynamic, 1 >::Zero(solution.getNumItems());
	createWeigthVector(itemId, weightVec);
	currrentPieceMap->setDataFromMatrix(overlapMatrix, weightVec);
	return currrentPieceMap;
}
