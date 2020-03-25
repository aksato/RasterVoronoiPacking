#ifndef TOTALOVERLAPMAPCUDA_H
#define TOTALOVERLAPMAPCUDA_H

#include "raster/rasternofitpolygon.h"
#include "raster/rasterstrippackingparameters.h"
#include "raster/totaloverlapmap.h"

namespace RASTERVORONOIPACKING {

	class TotalOverlapMapCuda : public TotalOverlapMap
	{
	public:
		TotalOverlapMapCuda(std::shared_ptr<RasterNoFitPolygon> ifp, int _cuttingStockLength = -1);
		TotalOverlapMapCuda(int width, int height, QPoint _reference, int _cuttingStockLength = -1);
		TotalOverlapMapCuda(QRect &boundingBox, int _cuttingStockLength = -1);
		~TotalOverlapMapCuda();

		void initCuda(uint _width, uint _height);

		void reset();
		void setDimensions(int _width, int _height);
		void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos);
		void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, int weight);
		void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, int weight, int zoomFactorInt) {}
		void changeTotalItems(int _totalNumItems) {} // FIXME: Better way to evaluate partial cached overlap map
		quint32 getMinimum(QPoint& minPt);
	};
}

#endif // TOTALOVERLAPMAPCUDA_H
