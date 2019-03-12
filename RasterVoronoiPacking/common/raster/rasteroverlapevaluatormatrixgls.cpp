#include "rasteroverlapevaluatormatrixgls.h"
#include "totaloverlapmatrix.h"
#include <Eigen/Core>

using namespace RASTERVORONOIPACKING;

RasterTotalOverlapMapEvaluatorMatrixGLS::RasterTotalOverlapMapEvaluatorMatrixGLS(std::shared_ptr<RasterPackingProblem> _problem, bool cuttingStock) : RasterTotalOverlapMapEvaluator(_problem, cuttingStock), matrices(_problem->count()) {
	for (int itemId = 0; itemId < problem->count(); itemId++) {
		for (uint angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
			std::shared_ptr<RasterNoFitPolygon> ifp = problem->getIfps()->getRasterNoFitPolygon(0, 0, problem->getItemType(itemId), angle);
			std::shared_ptr<TotalOverlapMatrix> curMap = std::shared_ptr<TotalOverlapMatrix>(new TotalOverlapMatrix(ifp->width(), ifp->height(), ifp->getOrigin(), _problem->count(), -1));
			matrices.addOverlapMap(itemId, angle, curMap);
			// FIXME: Delete innerift polygons as they are used to release memomry
		}
	}
}

RasterTotalOverlapMapEvaluatorMatrixGLS::RasterTotalOverlapMapEvaluatorMatrixGLS(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<GlsWeightSet> _glsWeights, bool cuttingStock) : RasterTotalOverlapMapEvaluator(_problem, _glsWeights, cuttingStock), matrices(_problem->count()) {
	for (int itemId = 0; itemId < problem->count(); itemId++) {
		for (uint angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
			std::shared_ptr<RasterNoFitPolygon> ifp = problem->getIfps()->getRasterNoFitPolygon(0, 0, problem->getItemType(itemId), angle);
			std::shared_ptr<TotalOverlapMatrix> curMap = std::shared_ptr<TotalOverlapMatrix>(new TotalOverlapMatrix(ifp->width(), ifp->height(), ifp->getOrigin(), _problem->count(), -1));
			matrices.addOverlapMap(itemId, angle, curMap);
			// FIXME: Delete innerift polygons as they are used to release memomry
		}
	}
}

void RasterTotalOverlapMapEvaluatorMatrixGLS::updateMapsLength(int pixelWidth) {
	for (int itemId = 0; itemId < problem->count(); itemId++)
		for (uint angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
			std::shared_ptr<TotalOverlapMatrix> curMatrix = matrices.getOverlapMap(itemId, angle);
			curMatrix->setRelativeWidth(problem->getContainerWidth() - pixelWidth);
		}
}

void RasterTotalOverlapMapEvaluatorMatrixGLS::createWeigthVector(int itemId, Eigen::Matrix< unsigned int, Eigen::Dynamic, 1 > &vec) {
	for (int i = 0; i < problem->count(); i++) {
		if (i == itemId) continue;
		vec.coeffRef(i) = getWeight(itemId, i);
	}
}

std::shared_ptr<TotalOverlapMap> RasterTotalOverlapMapEvaluatorMatrixGLS::getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution) {
	std::shared_ptr<TotalOverlapMatrix> currrentPieceMat = matrices.getOverlapMap(itemId, orientation);
	currrentPieceMat->reset();

	std::shared_ptr<ItemRasterNoFitPolygonSet> curItemNfpSet = problem->getNfps()->getItemRasterNoFitPolygonSet(problem->getItemType(itemId), orientation);
	for (int i = 0; i < problem->count(); i++) {
		if (i == itemId) continue;
		currrentPieceMat->addVoronoi(i, curItemNfpSet->getRasterNoFitPolygon(problem->getItemType(i), solution.getOrientation(i)), solution.getPosition(i));
	}
	
	Eigen::Matrix< unsigned int, Eigen::Dynamic, 1 >  weightVec = Eigen::Matrix< unsigned int, Eigen::Dynamic, 1 >::Zero(solution.getNumItems());
	createWeigthVector(itemId, weightVec);
	std::shared_ptr<TotalOverlapMap> currrentPieceMap = std::shared_ptr<TotalOverlapMap>(new TotalOverlapMap(currrentPieceMat->getRect(), currrentPieceMat->getCuttingStockLength()));
	Eigen::Matrix< unsigned int, Eigen::Dynamic, 1 > overlapMapMat = currrentPieceMat->getMatrixRef() * weightVec;
	currrentPieceMap->copyData(overlapMapMat.data());
	return currrentPieceMap;
}
