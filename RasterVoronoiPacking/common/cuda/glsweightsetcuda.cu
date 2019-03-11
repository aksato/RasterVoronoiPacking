#include "glsweightsetcuda.h"
#include "raster/rasterpackingproblem.h"
#include <QImage>
#include <QDebug>
#include "colormap.h"

#define INITWEIGHTVAL 100
using namespace RASTERVORONOIPACKING;

GlsWeightSetCuda::~GlsWeightSetCuda() {
	if (cudaWeights) cudaFree(cudaWeights);
}

void GlsWeightSetCuda::initCuda(int numItems) {
	//weights = QVector<unsigned int>(numItems * numItems);
	//weights.fill(INITWEIGHTVAL);
	cudaMalloc((void **)&cudaWeights, numItems * numItems * sizeof(unsigned int));
	cudaMemset(cudaWeights, INITWEIGHTVAL, numItems * numItems * sizeof(unsigned int));
}

void GlsWeightSetCuda::addWeight(int itemId1, int itemId2, int weight) {
    Q_ASSERT_X(itemId1 != itemId2, "GlsWeightSet::getWeigth", "Cannot verify collision with itself");
	weights[itemId1 + numItems*itemId2] = weight;
	weights[itemId2 + numItems*itemId1] = weight;
}

void GlsWeightSetCuda::updateWeights(QVector<WeightIncrement> &increments) {
    std::for_each(increments.begin(), increments.end(), [this](WeightIncrement &inc){
		weights[inc.id1 + numItems*inc.id2] += inc.value;
		weights[inc.id2 + numItems*inc.id1] += inc.value;
    });
}

void GlsWeightSetCuda::updateCudaWeights() {
	cudaMemcpy(cudaWeights, weights.data(), numItems * numItems * sizeof(unsigned int), cudaMemcpyHostToDevice);
}

unsigned int *GlsWeightSetCuda::getCudaWeights(int itemId) {
	return cudaWeights + numItems * itemId;
}