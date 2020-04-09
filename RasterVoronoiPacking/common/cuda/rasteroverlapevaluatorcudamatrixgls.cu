#include "cuda/rasteroverlapevaluatorcudamatrixgls.h"
#include "cuda/totaloverlapmatrixcuda.h"

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
			if(e + BLOCK_SIZE * m < nCols && tid < nRows) y_val += dA[tid + (e + BLOCK_SIZE * m) * nRows] * x_shared[e];
		}

		__syncthreads();
	}

	if (tid < nRows) dy[tid] = y_val;
}

RasterTotalOverlapMapEvaluatorCudaMatrixGLS::RasterTotalOverlapMapEvaluatorCudaMatrixGLS(std::shared_ptr<RasterPackingProblem> _problem, bool cuttingStock) : RasterTotalOverlapMapEvaluatorCudaGLS(_problem), matrices(_problem->count()) {
	glsWeightsCuda = std::shared_ptr<GlsWeightSetCuda>(new GlsWeightSetCuda(problem->count()));
	streams = std::vector<cudaStream_t>(problem->count());
	for (int i = 0; i < problem->count(); i++) cudaStreamCreate(&streams[i]);
	populateMatrices();
}

RasterTotalOverlapMapEvaluatorCudaMatrixGLS::RasterTotalOverlapMapEvaluatorCudaMatrixGLS(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<GlsWeightSetCuda> _glsWeightsCuda, bool cuttingStock) : RasterTotalOverlapMapEvaluatorCudaGLS(_problem, _glsWeightsCuda), glsWeightsCuda(_glsWeightsCuda), matrices(_problem->count()) {
	streams = std::vector<cudaStream_t>(problem->count());
	for (int i = 0; i < problem->count(); i++) cudaStreamCreate(&streams[i]);
	populateMatrices();
}

void RasterTotalOverlapMapEvaluatorCudaMatrixGLS::populateMatrices() {
	for (int itemTypeId = 0; itemTypeId < problem->getItemTypeCount(); itemTypeId++) {
		for (uint angle = 0; angle < (*problem->getItemByType(itemTypeId))->getAngleCount(); angle++) {
			std::shared_ptr<RasterNoFitPolygon> ifp = problem->getIfps()->getRasterNoFitPolygon(0, 0, itemTypeId, angle);
			std::shared_ptr<TotalOverlapMatrixCuda> curMap = std::shared_ptr<TotalOverlapMatrixCuda>(new TotalOverlapMatrixCuda(ifp->width(), ifp->height(), ifp->getOrigin(), problem->count(), streams, -1));
			matrices.addOverlapMap(itemTypeId, angle, curMap);
			// FIXME: Delete innerift polygons as they are used to release memomry
		}
	}
}

void RasterTotalOverlapMapEvaluatorCudaMatrixGLS::updateMapsLength(int pixelWidth) {
	RasterTotalOverlapMapEvaluatorCudaGLS::updateMapsLength(pixelWidth);
	for (int itemId = 0; itemId < problem->count(); itemId++)
		for (uint angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
			std::shared_ptr<TotalOverlapMatrixCuda> curMatrix = matrices.getOverlapMap(problem->getItemType(itemId), angle);
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
	std::shared_ptr<TotalOverlapMatrixCuda> currrentPieceMat = matrices.getOverlapMap(problem->getItemType(itemId), orientation);
	currrentPieceMat->reset();

	cudaDeviceSynchronize();
	std::shared_ptr<ItemRasterNoFitPolygonSet> curItemNfpSet = problem->getNfps()->getItemRasterNoFitPolygonSet(problem->getItemType(itemId), orientation);
	for (int i = 0; i < problem->count(); i++) {
		if (i == itemId) continue;
		currrentPieceMat->addVoronoi(i, curItemNfpSet->getRasterNoFitPolygon(problem->getItemType(i), solution.getOrientation(i)), solution.getPosition(i)); 
	}
	std::shared_ptr<TotalOverlapMap> currrentPieceMap = cudamaps.getOverlapMap(itemId, orientation);
	dim3 dim_grid((currrentPieceMat->getHeight()* currrentPieceMat->getWidth() + BLOCK_SIZE - 1) / BLOCK_SIZE);
	dim3 dim_block(BLOCK_SIZE);
	cudaDeviceSynchronize();
	gemv << <dim_grid, dim_block >> > (currrentPieceMat->getData(), glsWeightsCuda->getCudaWeights(itemId), currrentPieceMap->getData() , currrentPieceMat->getHeight() * currrentPieceMat->getWidth(), solution.getNumItems());
	cudaDeviceSynchronize();

	return currrentPieceMap;
}