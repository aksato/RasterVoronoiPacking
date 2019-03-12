#ifndef TOTALOVERLAPMATRIX_H
#define TOTALOVERLAPMATRIX_H

#include "raster/rasternofitpolygon.h"
#include "raster/rasterstrippackingparameters.h"
#include "raster/totaloverlapmap.h"
#include <Eigen/Core>

namespace RASTERVORONOIPACKING {

	class TotalOverlapMatrix : public TotalOverlapMap
	{
	public:
		TotalOverlapMatrix(std::shared_ptr<RasterNoFitPolygon> ifp, int _numItems, int _cuttingStockLength = -1);
		TotalOverlapMatrix(int width, int height, QPoint _reference, int _numItems, int _cuttingStockLength = -1);
		TotalOverlapMatrix(QRect &boundingBox, int _numItems, int _cuttingStockLength = -1);
		~TotalOverlapMatrix() { data = nullptr; }

		void initMatrix(uint _width, uint _height);

		void reset();
		void setDimensions(int _width, int _height);
		void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos);
		void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, int weight) {}
		void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, int weight, int zoomFactorInt) {}
		void changeTotalItems(int _totalNumItems) {} // FIXME: Better way to evaluate partial cached overlap map

		Eigen::Matrix< unsigned int, Eigen::Dynamic, Eigen::Dynamic > &getMatrixRef() { return matrixData;}
	private:
		int numItems;
		Eigen::Matrix< unsigned int, Eigen::Dynamic, Eigen::Dynamic > matrixData;
	};

}

#endif // TOTALOVERLAPMATRIX_H
