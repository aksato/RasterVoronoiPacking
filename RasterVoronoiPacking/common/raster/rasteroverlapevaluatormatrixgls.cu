#include "rasteroverlapevaluatormatrixgls.h"

using namespace RASTERVORONOIPACKING;

std::shared_ptr<TotalOverlapMap> RasterTotalOverlapMapEvaluatorCudaMatrixGLS::getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution) {
	std::shared_ptr<TotalOverlapMap> currrentPieceMap = maps.getOverlapMap(itemId, orientation);
	Eigen::Matrix< unsigned int, Eigen::Dynamic, Eigen::Dynamic > overlapMatrix = Eigen::Matrix< unsigned int, Eigen::Dynamic, Eigen::Dynamic >::Zero(currrentPieceMap->getHeight() * currrentPieceMap->getWidth(), problem->count());

	// Alloc current overlap map on GPU
	auto deleter = [&](quint32* ptr) { cudaFree(ptr); };
	std::shared_ptr<quint32> d_currrentPieceMapData(new quint32[currrentPieceMap->getHeight()*currrentPieceMap->getWidth()], deleter);
	cudaMalloc((void **)&d_currrentPieceMapData, currrentPieceMap->getHeight()*currrentPieceMap->getWidth() * sizeof(quint32));
	cudaMemcpy(d_currrentPieceMapData.get(), currrentPieceMap->getData(), currrentPieceMap->getHeight()*currrentPieceMap->getWidth() * sizeof(quint32), cudaMemcpyHostToDevice);

	// Alloc matrix on GPU
	int overlapMatrixLength = currrentPieceMap->getHeight() * currrentPieceMap->getWidth() * problem->count();
	std::shared_ptr<quint32> d_overlapMatrix(new quint32[overlapMatrixLength], deleter);
	cudaMalloc((void **)&d_overlapMatrix, overlapMatrixLength * sizeof(quint32));
	cudaMemset(d_overlapMatrix.get(), 0, overlapMatrixLength * sizeof(quint32));

	std::shared_ptr<ItemRasterNoFitPolygonSet> curItemNfpSet = problem->getNfps()->getItemRasterNoFitPolygonSet(problem->getItemType(itemId), orientation);
	for (int i = 0; i < problem->count(); i++) {
		if (i == itemId) continue;

		// Alloc current nofit polygon on GPU
		std::shared_ptr<RasterNoFitPolygon> currentRasterNoFitPolygon = curItemNfpSet->getRasterNoFitPolygon(problem->getItemType(i), solution.getOrientation(i));
		std::shared_ptr<quint32> d_currentRasterNoFitPolygon(new quint32[currentRasterNoFitPolygon->height() * currentRasterNoFitPolygon->width()], deleter);
		cudaMalloc((void **)&d_currentRasterNoFitPolygon, currentRasterNoFitPolygon->height() * currentRasterNoFitPolygon->width() * sizeof(quint32));
		cudaMemcpy(d_currentRasterNoFitPolygon.get(), currentRasterNoFitPolygon->getPixelRef(0,0), currentRasterNoFitPolygon->height() * currentRasterNoFitPolygon->width() * sizeof(quint32), cudaMemcpyHostToDevice);

		// Add nfp to overlap map
		currrentPieceMap->addToMatrixCuda(i, d_currentRasterNoFitPolygon, currentRasterNoFitPolygon->getOrigin(), currentRasterNoFitPolygon->width(), currentRasterNoFitPolygon->height(), solution.getPosition(i), d_overlapMatrix, d_currrentPieceMapData);
	}
	cudaMemcpy(overlapMatrix.data(), d_overlapMatrix.get(), overlapMatrixLength * sizeof(quint32), cudaMemcpyDeviceToHost);

	Eigen::Matrix< unsigned int, Eigen::Dynamic, 1 >  weightVec = Eigen::Matrix< unsigned int, Eigen::Dynamic, 1 >::Zero(solution.getNumItems());
	createWeigthVector(itemId, weightVec);
	currrentPieceMap->setDataFromMatrix(overlapMatrix, weightVec);
	return currrentPieceMap;
}