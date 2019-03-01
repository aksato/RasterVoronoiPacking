#include "rasteroverlapevaluatormatrixgls.h"

using namespace RASTERVORONOIPACKING;

RasterTotalOverlapMapEvaluatorCudaMatrixGLS::RasterTotalOverlapMapEvaluatorCudaMatrixGLS(std::shared_ptr<RasterPackingProblem> _problem, bool cuttingStock) : RasterTotalOverlapMapEvaluatorMatrixGLSBase(_problem, cuttingStock), matrices(_problem->count()) {
	for (int itemId = 0; itemId < problem->count(); itemId++) {
		for (uint angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
			std::shared_ptr<RasterNoFitPolygon> ifp = problem->getIfps()->getRasterNoFitPolygon(0, 0, problem->getItemType(itemId), angle);
			std::shared_ptr<TotalOverlapMatrixCuda> curMap = std::shared_ptr<TotalOverlapMatrixCuda>(new TotalOverlapMatrixCuda(ifp->width() , ifp->height(), ifp->getOrigin(), _problem->count(), -1));
			matrices.addOverlapMap(itemId, angle, curMap);
			// FIXME: Delete innerift polygons as they are used to release memomry
		}
	}
}

RasterTotalOverlapMapEvaluatorCudaMatrixGLS::RasterTotalOverlapMapEvaluatorCudaMatrixGLS(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<GlsWeightSet> _glsWeights, bool cuttingStock) : RasterTotalOverlapMapEvaluatorMatrixGLSBase(_problem, _glsWeights, cuttingStock), matrices(_problem->count()) {
	for (int itemId = 0; itemId < problem->count(); itemId++) {
		for (uint angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
			std::shared_ptr<RasterNoFitPolygon> ifp = problem->getIfps()->getRasterNoFitPolygon(0, 0, problem->getItemType(itemId), angle);
			std::shared_ptr<TotalOverlapMatrixCuda> curMap = std::shared_ptr<TotalOverlapMatrixCuda>(new TotalOverlapMatrixCuda(ifp->width() , ifp->height(), ifp->getOrigin(), _problem->count(), -1));
			matrices.addOverlapMap(itemId, angle, curMap);
			// FIXME: Delete innerift polygons as they are used to release memomry
		}
	}
}

void RasterTotalOverlapMapEvaluatorCudaMatrixGLS::updateMapsLength(int pixelWidth) {
	RasterTotalOverlapMapEvaluatorGLS::updateMapsLength(pixelWidth);
	for (int itemId = 0; itemId < problem->count(); itemId++)
		for (uint angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
			std::shared_ptr<TotalOverlapMatrixCuda> curMatrix = matrices.getOverlapMap(itemId, angle);
			curMatrix->setRelativeWidth(problem->getContainerWidth() - pixelWidth);
		}
}

std::shared_ptr<TotalOverlapMap> RasterTotalOverlapMapEvaluatorCudaMatrixGLS::getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution) {
	std::shared_ptr<TotalOverlapMatrixCuda> currrentPieceMat = matrices.getOverlapMap(itemId, orientation);
	currrentPieceMat->reset();

	std::shared_ptr<ItemRasterNoFitPolygonSet> curItemNfpSet = problem->getNfps()->getItemRasterNoFitPolygonSet(problem->getItemType(itemId), orientation);
	for (int i = 0; i < problem->count(); i++) {
		if (i == itemId) continue;

		// Add nfp to overlap map
		currrentPieceMat->addVoronoi(i, curItemNfpSet->getRasterNoFitPolygon(problem->getItemType(i), solution.getOrientation(i)), solution.getPosition(i)); 
	}
	Eigen::Matrix< unsigned int, Eigen::Dynamic, Eigen::Dynamic > overlapMatrix = Eigen::Matrix< unsigned int, Eigen::Dynamic, Eigen::Dynamic >::Zero(currrentPieceMat->getHeight() * currrentPieceMat->getWidth(), problem->count());
	cudaMemcpy(overlapMatrix.data(), currrentPieceMat->getData(), currrentPieceMat->getHeight() * currrentPieceMat->getWidth() * problem->count() * sizeof(quint32), cudaMemcpyDeviceToHost);
	
	Eigen::Matrix< unsigned int, Eigen::Dynamic, 1 >  weightVec = Eigen::Matrix< unsigned int, Eigen::Dynamic, 1 >::Zero(solution.getNumItems());
	createWeigthVector(itemId, weightVec);
	std::shared_ptr<TotalOverlapMap> currrentPieceMap = maps.getOverlapMap(itemId, orientation);
	//currrentPieceMap->setDataFromMatrix(overlapMatrix, weightVec);
	Eigen::Matrix< unsigned int, Eigen::Dynamic, 1 > overlapMapMat = overlapMatrix * weightVec;
	currrentPieceMap->copyData(overlapMapMat.data());
	return currrentPieceMap;
}