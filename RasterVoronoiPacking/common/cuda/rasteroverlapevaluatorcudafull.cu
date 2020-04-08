#include "rasteroverlapevaluatorcudafull.h"

#define THREADS_PER_BLOCK 512
typedef quint32* NfpData;
using namespace RASTERVORONOIPACKING;

__device__
int getRasterNoFitPolygonKey(int staticPieceTypeId, int staticAngleId, int orbitingPieceTypeId, int orbitingAngleId, int numAngles, int numKeys) {
	int staticKey = staticPieceTypeId * numAngles + staticAngleId;
	int orbitingKey = orbitingPieceTypeId * numAngles + orbitingAngleId;
	return staticKey + numKeys * orbitingKey;
}

__device__
int getWeight(unsigned int* weights, int itemId1, int itemId2, int numItems) {
	if (itemId1 > itemId2) return weights[itemId1 + numItems * itemId2];
	return weights[itemId2 + numItems * itemId1];
}

__global__
void cudaGetTotalOverlapMap(quint32* map, int width, int height, int referencePointX, int referencePointY, DeviceRasterNoFitPolygonSet nfps, DeviceRasterPackingSolution solution,
	int numItems, int *itemType, int itemId, int orientation, int numAngles, int numKeys, unsigned int *weights) {
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i >= width || j >= height)
		return;

	for (int k = 0; k < numItems; k++) {
	//for (int k = 18; k < 19; k++) {
		if (k == itemId) continue;
		int nfpKey = getRasterNoFitPolygonKey(itemType[k], solution.d_orientations[k], itemType[itemId], orientation, numAngles, numKeys);
		NfpData nfp = nfps.d_data[nfpKey];
		int nfpWidth = nfps.d_widths[nfpKey]; int nfpHeight = nfps.d_heights[nfpKey];
		int nfpOriginX = nfps.d_originsX[nfpKey]; int nfpOriginY = nfps.d_originsY[nfpKey];
		int nfpMutiplier = nfps.d_multipliers[nfpKey];
		int relativeOriginX = referencePointX + solution.d_posX[k] - nfpMutiplier * nfpOriginX + (nfpWidth - 1) * (nfpMutiplier - 1) / 2;
		int relativeOriginY = referencePointY + solution.d_posY[k] - nfpMutiplier * nfpOriginY + (nfpHeight - 1) * (nfpMutiplier - 1) / 2;

		if (i < relativeOriginX || i > relativeOriginX + nfpWidth - 1 || j < relativeOriginY || j > relativeOriginY + nfpHeight - 1) {}
		else {
			int nfpidx = nfpMutiplier < 0 ? 
				nfpWidth * nfpHeight - 1 - j + relativeOriginY - (i - relativeOriginX) * nfpHeight :
				j - relativeOriginY + (i - relativeOriginX) * nfpHeight;
			map[i * height + j] += nfp[nfpidx] * getWeight(weights, itemId, k, numItems);
		}
	}
}

RasterTotalOverlapMapEvaluatorCudaFull::RasterTotalOverlapMapEvaluatorCudaFull(std::shared_ptr<RasterPackingProblem> _problem) :
	RasterTotalOverlapMapEvaluatorCudaGLS(_problem) {
	glsWeightsCuda = std::shared_ptr<GlsWeightSetCuda>(new GlsWeightSetCuda(_problem->count()));
	initCuda(_problem);
}

RasterTotalOverlapMapEvaluatorCudaFull::RasterTotalOverlapMapEvaluatorCudaFull(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<GlsWeightSetCuda> _glsWeightsCuda) : 
	RasterTotalOverlapMapEvaluatorCudaGLS(_problem, _glsWeightsCuda), glsWeightsCuda(_glsWeightsCuda) {
	initCuda(_problem);
}

RasterTotalOverlapMapEvaluatorCudaFull::~RasterTotalOverlapMapEvaluatorCudaFull() {
	cudaFree(d_nfps.d_data);
	cudaFree(d_nfps.d_widths);
	cudaFree(d_nfps.d_heights);
	cudaFree(d_nfps.d_originsX);
	cudaFree(d_nfps.d_originsY);
	cudaFree(d_nfps.d_multipliers);
	cudaFree(d_itemId2ItemTypeMap);
}

void RasterTotalOverlapMapEvaluatorCudaFull::initCuda(std::shared_ptr<RasterPackingProblem> _problem) {
	numKeys = _problem->getNfps()->getSize();
	numAngles = problem->getNfps()->getNumAngles();

	NfpData* data;
	int* widths, * heights;
	int* originsX, * originsY, * multipliers;
	data = new NfpData[numKeys * numKeys];
	widths = new int[numKeys * numKeys]; heights = new int[numKeys * numKeys];
	originsX = new int[numKeys * numKeys]; originsY = new int[numKeys * numKeys];
	multipliers = new int[numKeys * numKeys];

	for (int itemTypeId1 = 0; itemTypeId1 < problem->getItemTypeCount(); itemTypeId1++) {
		for (uint angle1 = 0; angle1 < (*problem->getItemByType(itemTypeId1))->getAngleCount(); angle1++) {
			for (int itemTypeId2 = 0; itemTypeId2 < problem->getItemTypeCount(); itemTypeId2++) {
				for (uint angle2 = 0; angle2 < (*problem->getItemByType(itemTypeId2))->getAngleCount(); angle2++) {
					int staticKey = itemTypeId1 * numAngles + angle1;
					int orbitingKey = itemTypeId2 * numAngles + angle2;
					std::shared_ptr<RasterNoFitPolygon> curNfp = problem->getNfps()->getRasterNoFitPolygon(itemTypeId1, angle1, itemTypeId2, angle2);
					data[staticKey + numKeys * orbitingKey] = curNfp->getMatrix();
					widths[staticKey + numKeys * orbitingKey] = curNfp->width();
					heights[staticKey + numKeys * orbitingKey] = curNfp->height();
					originsX[staticKey + numKeys * orbitingKey] = curNfp->getOrigin().x();
					originsY[staticKey + numKeys * orbitingKey] = curNfp->getOrigin().y();
					multipliers[staticKey + numKeys * orbitingKey] = curNfp->getFlipMultiplier();
				}
			}
		}
	}

	//allocation
	cudaMalloc((void**)&d_nfps.d_data, (numKeys * numKeys) * sizeof(NfpData));
	cudaMalloc((void**)&d_nfps.d_widths, (numKeys * numKeys) * sizeof(int));
	cudaMalloc((void**)&d_nfps.d_heights, (numKeys * numKeys) * sizeof(int));
	cudaMalloc((void**)&d_nfps.d_originsX, (numKeys * numKeys) * sizeof(int));
	cudaMalloc((void**)&d_nfps.d_originsY, (numKeys * numKeys) * sizeof(int));
	cudaMalloc((void**)&d_nfps.d_multipliers, (numKeys * numKeys) * sizeof(int));

	//copying from host to device
	cudaMemcpy(d_nfps.d_data, data, (numKeys * numKeys) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nfps.d_widths, widths, (numKeys * numKeys) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nfps.d_heights, heights, (numKeys * numKeys) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nfps.d_originsX, originsX, (numKeys * numKeys) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nfps.d_originsY, originsY, (numKeys * numKeys) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nfps.d_multipliers, multipliers, (numKeys * numKeys) * sizeof(int), cudaMemcpyHostToDevice);


	int* itemId2ItemTypeMap = new int[_problem->count()];
	for (int k = 0; k < _problem->count(); k++)
		itemId2ItemTypeMap[k] = _problem->getItemType(k);
	cudaMalloc((void**)&d_itemId2ItemTypeMap, _problem->count() * sizeof(int));
	cudaMemcpy(d_itemId2ItemTypeMap, itemId2ItemTypeMap, _problem->count() * sizeof(int), cudaMemcpyHostToDevice);

	delete[] data;
	delete[] widths;
	delete[] heights;
	delete[] originsX;
	delete[] originsY;
	delete[] multipliers;
	delete[] itemId2ItemTypeMap;
}

void RasterTotalOverlapMapEvaluatorCudaFull::updateWeights(RasterPackingSolution &solution) {
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

void RasterTotalOverlapMapEvaluatorCudaFull::updateWeights(RasterPackingSolution &solution, QVector<quint32> &overlaps, quint32 maxOverlap) {
	std::transform(glsWeightsCuda->begin(), glsWeightsCuda->end(), overlaps.begin(),
		glsWeightsCuda->begin(), [&maxOverlap](const quint32 &a, const quint32 &b) {return a + qRound(100.0*(qreal)b / (qreal)maxOverlap); });
	// Update on GPU
	glsWeightsCuda->updateCudaWeights();
}


//  TODO: Update cache information!
void RasterTotalOverlapMapEvaluatorCudaFull::resetWeights() {
	glsWeightsCuda->reset(problem->count());
}

// Determines the item total overlap map for a given orientation in a solution
std::shared_ptr<TotalOverlapMap> RasterTotalOverlapMapEvaluatorCudaFull::getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution& solution) {
	std::shared_ptr<TotalOverlapMap> currrentPieceMap = cudamaps.getOverlapMap(itemId, orientation);
	currrentPieceMap->reset();

	DeviceRasterPackingSolution d_solution;
	int* posX = new int[problem->count()];
	int* posY = new int[problem->count()];
	int *orientations = new int[problem->count()];
	for (int k = 0; k < problem->count(); k++) {
		posX[k] = solution.getPosition(k).x();
		posY[k] = solution.getPosition(k).y();
		orientations[k] = solution.getOrientation(k);
	}
	cudaMalloc((void**)&d_solution.d_posX, (problem->count()) * sizeof(int));
	cudaMalloc((void**)&d_solution.d_posY, (problem->count()) * sizeof(int));
	cudaMalloc((void**)&d_solution.d_orientations, (problem->count()) * sizeof(int));
	cudaMemcpy(d_solution.d_posX, posX, problem->count() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_solution.d_posY, posY, problem->count() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_solution.d_orientations, orientations, problem->count() * sizeof(int), cudaMemcpyHostToDevice);
	delete[] posX;
	delete[] posY;
	delete[] orientations;

	int numBlocksX = (currrentPieceMap->getWidth() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	int numBlocksY = (currrentPieceMap->getHeight() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	dim3 numBlocks(numBlocksX, numBlocksY);
	dim3 numThreads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
	cudaDeviceSynchronize();
	cudaGetTotalOverlapMap <<< numThreads, numBlocks >>> (
		currrentPieceMap->getData(), currrentPieceMap->getWidth(), currrentPieceMap->getHeight(), currrentPieceMap->getReferencePoint().x(), currrentPieceMap->getReferencePoint().y(),
		d_nfps, d_solution, problem->count(), d_itemId2ItemTypeMap, itemId, orientation, numAngles, numKeys, glsWeightsCuda->getCudaWeights(0));

	cudaFree(d_solution.d_posX);
	cudaFree(d_solution.d_posY);
	cudaFree(d_solution.d_orientations);
	
	return currrentPieceMap;
}