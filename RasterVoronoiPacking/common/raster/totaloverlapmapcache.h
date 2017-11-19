#ifndef TOTALOVERLAPMAPCACHE_H
#define TOTALOVERLAPMAPCACHE_H

#include "rasternofitpolygon.h"
#include "rasterstrippackingparameters.h"
#include "totaloverlapmap.h"

namespace RASTERVORONOIPACKING {

	class CachedTotalOverlapMap : public TotalOverlapMap {
	public:
		CachedTotalOverlapMap(std::shared_ptr<RasterNoFitPolygon> ifp, int _totalNumItems) : TotalOverlapMap(ifp), totalNumItems(_totalNumItems), currentCount(0) { };
		CachedTotalOverlapMap(int width, int height, QPoint _reference, int _totalNumItems) : TotalOverlapMap(width, height, _reference), totalNumItems(_totalNumItems), currentCount(0) { };
		~CachedTotalOverlapMap() {}

		void setDimensions(int _width, int _height);
		void reset() {}
		void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos);
		void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, int weight);
		void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, int weight, int zoomFactorInt);
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
		const int totalNumItems;
	};
}

#endif // TOTALOVERLAPMAPCACHE_H
