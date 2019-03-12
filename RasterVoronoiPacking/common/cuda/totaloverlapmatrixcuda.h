#ifndef TOTALOVERLAPMATRIXCUDA_H
#define TOTALOVERLAPMATRIXCUDA_H

#include <cuda_runtime.h>
#include "raster/rasternofitpolygon.h"
#include "raster/rasterstrippackingparameters.h"
#include "raster/totaloverlapmatrix.h"

namespace RASTERVORONOIPACKING {

	class TotalOverlapMatrixCuda : public TotalOverlapMap
	{
	public:
		TotalOverlapMatrixCuda(std::shared_ptr<RasterNoFitPolygon> ifp, int _numItems, std::vector<cudaStream_t> &_streams, int _cuttingStockLength = -1);
		TotalOverlapMatrixCuda(int width, int height, QPoint _reference, int _numItems, std::vector<cudaStream_t> &_streams, int _cuttingStockLength = -1);
		TotalOverlapMatrixCuda(QRect &boundingBox, int _numItems, std::vector<cudaStream_t> &_streams, int _cuttingStockLength = -1);
		~TotalOverlapMatrixCuda();

		void initCuda(uint _width, uint _height);

		void reset();
		void setDimensions(int _width, int _height);
		void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos);
		void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, int weight) {}
		void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, int weight, int zoomFactorInt) {}
		void changeTotalItems(int _totalNumItems) {} // FIXME: Better way to evaluate partial cached overlap map

	private:
		int numItems;
		std::vector<cudaStream_t> &streams;
	};

}

#endif // TOTALOVERLAPMATRIXCUDA_H
