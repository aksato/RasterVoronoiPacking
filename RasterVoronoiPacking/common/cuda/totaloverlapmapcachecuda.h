#ifndef TOTALOVERLAPMAPCACHECUDA_H
#define TOTALOVERLAPMAPCACHECUDA_H

#include "raster/rasternofitpolygon.h"
#include "raster/rasterstrippackingparameters.h"
#include "totaloverlapmapcuda.h"

namespace RASTERVORONOIPACKING {

	class CachedTotalOverlapMapCuda : public TotalOverlapMapCuda {
	public:
		CachedTotalOverlapMapCuda(std::shared_ptr<RasterNoFitPolygon> ifp, int _cuttingStockLength, int _totalNumItems) : TotalOverlapMapCuda(ifp, _cuttingStockLength), totalNumItems(_totalNumItems), currentCount(0) { };
		CachedTotalOverlapMapCuda(int width, int height, QPoint _reference, int _cuttingStockLength, int _totalNumItems) : TotalOverlapMapCuda(width, height, _reference, _cuttingStockLength), totalNumItems(_totalNumItems), currentCount(0) { };
		~CachedTotalOverlapMapCuda() {}

		void setDimensions(int _width, int _height);
		void reset() {}
		void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos);
		void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, int weight);
		void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, int weight, int zoomFactorInt);
		void changeTotalItems(int _totalNumItems) { this->totalNumItems = _totalNumItems; }
	private:
		struct TotalOverlapMapEntry {
			TotalOverlapMapEntry() {}
			TotalOverlapMapEntry(std::shared_ptr<RasterNoFitPolygon> _nfp, QPoint pos, int _weight) : nfp(_nfp), posX(pos.x()), posY(pos.y()), weight(_weight) {}
			TotalOverlapMapEntry(TotalOverlapMapEntry &other, int _weight) : nfp(other.nfp), posX(other.posX), posY(other.posY), weight(_weight) {}
			std::shared_ptr<RasterNoFitPolygon> nfp;
			int posX, posY;
			int weight;
		};

		void processVoronoiEntry(int itemId, TotalOverlapMapEntry &currentEntry);
		void resetVoronoiEntriesCache();

		int currentCount;
		QList<TotalOverlapMapEntry> toAddEntries, toRemoveEntries;
		QMap<int, TotalOverlapMapEntry> currentEntries;
		int totalNumItems;
	};
}

#endif // TOTALOVERLAPMAPCACHECUDA_H
