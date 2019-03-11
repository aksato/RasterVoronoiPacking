#ifndef GLSWEIGHTSETCUDA_H
#define GLSWEIGHTSETCUDA_H

#include "raster/glsweightset.h"
#include <memory>
#include <QtGlobal>
#include <QHash>
#include <QVector>

class QImage;

namespace RASTERVORONOIPACKING {
    class RasterPackingSolution;
    class RasterPackingProblem;
	class GlsWeightSetCuda : public  GlsWeightSet {
	public:
		GlsWeightSetCuda(int _numItems) : GlsWeightSet(_numItems), cudaWeights(nullptr) {
			initCuda(_numItems);
		}
		~GlsWeightSetCuda();
		void initCuda(int numItems);
		void updateWeights(QVector<WeightIncrement> &increments);
		void addWeight(int itemId1, int itemId2, int weight);
		void updateCudaWeights();
		unsigned int *getCudaWeights(int itemId);
	private:
		unsigned int *cudaWeights;
	};
}

#endif // GLSWEIGHTSETCUDA_H
