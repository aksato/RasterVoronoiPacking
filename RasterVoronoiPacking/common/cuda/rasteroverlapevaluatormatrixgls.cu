#include "raster/rasteroverlapevaluatormatrixgls.h"

#define     BLOCK_SIZE      16

using namespace RASTERVORONOIPACKING;

__global__ void gemv(const quint32 * __restrict__ dA, const quint32 * __restrict__ dx, quint32 * __restrict__ dy, const unsigned int nRows, const unsigned int nCols)
{
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ quint32 x_shared[BLOCK_SIZE];

	quint32 y_val = 0;

#pragma unroll
	for (unsigned int m = 0; m < ((nCols + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
	{
		if ((m * BLOCK_SIZE + threadIdx.x) <  nCols) x_shared[threadIdx.x] = dx[threadIdx.x + m * BLOCK_SIZE];
		else                                         x_shared[threadIdx.x] = 0.f;
		__syncthreads();

#pragma unroll
		for (unsigned int e = 0; e < BLOCK_SIZE; ++e) {
			// --- Column-major ordering - faster
			y_val += dA[tid + (e + BLOCK_SIZE * m) * nRows] * x_shared[e];
			// --- Row-major ordering - slower
			//y_val += dA[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}

		__syncthreads();
	}

	if (tid < nRows) dy[tid] = y_val;
}

RasterTotalOverlapMapEvaluatorCudaMatrixGLS::RasterTotalOverlapMapEvaluatorCudaMatrixGLS(std::shared_ptr<RasterPackingProblem> _problem, bool cuttingStock) : RasterTotalOverlapMapEvaluator(_problem), matrices(_problem->count()) {
	glsWeightsCuda = std::shared_ptr<GlsWeightSetCuda>(new GlsWeightSetCuda(problem->count()));
	populateMaps();
}

RasterTotalOverlapMapEvaluatorCudaMatrixGLS::RasterTotalOverlapMapEvaluatorCudaMatrixGLS(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<GlsWeightSetCuda> _glsWeightsCuda, bool cuttingStock) : RasterTotalOverlapMapEvaluator(_problem, _glsWeightsCuda), glsWeightsCuda(_glsWeightsCuda), matrices(_problem->count()) {
	populateMaps();
}

void RasterTotalOverlapMapEvaluatorCudaMatrixGLS::populateMaps() {
	for (int itemId = 0; itemId < problem->count(); itemId++) {
		for (uint angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
			std::shared_ptr<RasterNoFitPolygon> ifp = problem->getIfps()->getRasterNoFitPolygon(0, 0, problem->getItemType(itemId), angle);
			std::shared_ptr<TotalOverlapMatrixCuda> curMap = std::shared_ptr<TotalOverlapMatrixCuda>(new TotalOverlapMatrixCuda(ifp->width(), ifp->height(), ifp->getOrigin(), problem->count(), -1));
			matrices.addOverlapMap(itemId, angle, curMap);
			// FIXME: Delete innerift polygons as they are used to release memomry
		}
	}
}

void RasterTotalOverlapMapEvaluatorCudaMatrixGLS::updateMapsLength(int pixelWidth) {
	for (int itemId = 0; itemId < problem->count(); itemId++)
		for (uint angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
			std::shared_ptr<TotalOverlapMatrixCuda> curMatrix = matrices.getOverlapMap(itemId, angle);
			curMatrix->setRelativeWidth(problem->getContainerWidth() - pixelWidth);
		}
}

void RasterTotalOverlapMapEvaluatorCudaMatrixGLS::updateWeights(RasterPackingSolution &solution) {
	QVector<WeightIncrement> solutionOverlapValues;

	// Determine pair overlap values
	for (int itemId1 = 0; itemId1 < problem->count(); itemId1++)
		for (int itemId2 = 0; itemId2 < problem->count(); itemId2++) {
			if (itemId1 == itemId2) continue;
			quint32 curOValue = problem->getDistanceValue(itemId1, solution.getPosition(itemId1), solution.getOrientation(itemId1),
				itemId2, solution.getPosition(itemId2), solution.getOrientation(itemId2));
			if (curOValue != 0) {
				solutionOverlapValues.append(WeightIncrement(itemId1, itemId2, 1));
			}
		}

	// Add to the current weight map
	glsWeightsCuda->updateWeights(solutionOverlapValues);
	// Update on GPU
	glsWeightsCuda->updateCudaWeights();
}

void RasterTotalOverlapMapEvaluatorCudaMatrixGLS::updateWeights(RasterPackingSolution &solution, QVector<quint32> &overlaps, quint32 maxOverlap) {
	std::transform(glsWeightsCuda->begin(), glsWeightsCuda->end(), overlaps.begin(),
		glsWeightsCuda->begin(), [&maxOverlap](const quint32 &a, const quint32 &b) {return a + qRound(100.0*(qreal)b / (qreal)maxOverlap); });
	// Update on GPU
	glsWeightsCuda->updateCudaWeights();
}

//  TODO: Update cache information!
void RasterTotalOverlapMapEvaluatorCudaMatrixGLS::resetWeights() {
	glsWeightsCuda->reset(problem->count());
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
	//Eigen::Matrix< unsigned int, Eigen::Dynamic, 1 >  weightVec = Eigen::Matrix< unsigned int, Eigen::Dynamic, 1 >::Zero(solution.getNumItems());
	//createWeigthVector(itemId, weightVec);
	//quint32 *weight; cudaMalloc((void **)&weight, solution.getNumItems() * sizeof(quint32));
	//cudaMemcpy(weight, weightVec.data(), solution.getNumItems() * sizeof(quint32), cudaMemcpyHostToDevice);
	quint32 *map;
	cudaMalloc((void **)&map, currrentPieceMat->getHeight() * currrentPieceMat->getWidth() * sizeof(quint32)); 
	cudaMemset(map, 0, currrentPieceMat->getHeight() * currrentPieceMat->getWidth() * sizeof(quint32));
	dim3 dim_grid((currrentPieceMat->getHeight()* currrentPieceMat->getWidth() + BLOCK_SIZE - 1) / BLOCK_SIZE);
	dim3 dim_block(BLOCK_SIZE);
	gemv << <dim_grid, dim_block >> >(currrentPieceMat->getData(), glsWeightsCuda->getCudaWeights(itemId), map, currrentPieceMat->getHeight() * currrentPieceMat->getWidth(), solution.getNumItems());
	//gemv << <dim_grid, dim_block >> >(currrentPieceMat->getData(), weight, map, currrentPieceMat->getHeight() * currrentPieceMat->getWidth(), solution.getNumItems());

	return std::shared_ptr<TotalOverlapMap>();
	//std::shared_ptr<TotalOverlapMap> dummyPieceMap = std::shared_ptr<TotalOverlapMap>(new TotalOverlapMap(currrentPieceMat->getRect(), currrentPieceMat->getCuttingStockLength()));
	//cudaMemcpy(dummyPieceMap->getData(), map, currrentPieceMat->getHeight() * currrentPieceMat->getWidth() * sizeof(quint32), cudaMemcpyDeviceToHost);
	//return dummyPieceMap;
}