#ifndef TOTALOVERLAPMAPCACHE_H
#define TOTALOVERLAPMAPCACHE_H

#include "rasternofitpolygon.h"
#include "rasterstrippackingparameters.h"
#include "totaloverlapmap.h"

namespace RASTERVORONOIPACKING {

	class CachedTotalOverlapMap : public TotalOverlapMap {
	public:
		CachedTotalOverlapMap(std::shared_ptr<RasterNoFitPolygon> ifp, int _totalNumItems) : TotalOverlapMap(ifp), totalNumItems(_totalNumItems), emptyCache(true), toRemoveCount(0), currentCount(0) { currentEntries.resize(totalNumItems); toRemoveEntries.resize(totalNumItems); };
		CachedTotalOverlapMap(int width, int height, QPoint _reference, int _totalNumItems) : TotalOverlapMap(width, height, _reference), totalNumItems(_totalNumItems), emptyCache(true), toRemoveCount(0), currentCount(0) { currentEntries.resize(totalNumItems); toRemoveEntries.resize(totalNumItems); };
		~CachedTotalOverlapMap() {}

		void init(uint _width, uint _height); // FIXME: Is it really necessary to reimplement?
		void reset() {}
		void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos);
		void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, int weight);
		void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, int weight, int zoomFactorInt);
	private:
		struct TotalOverlapMapEntry {
			TotalOverlapMapEntry() : enabled(false) {}
			TotalOverlapMapEntry(int _itemId, std::shared_ptr<RasterNoFitPolygon> _nfp, QPoint pos, int _weight) : itemId(_itemId), nfp(_nfp), posX(pos.x()), posY(pos.y()), weight(_weight), enabled(true) {}
			TotalOverlapMapEntry(TotalOverlapMapEntry &other, int _weight) : itemId(other.itemId), nfp(other.nfp), posX(other.posX), posY(other.posY), weight(_weight), enabled(other.enabled) {}
			int itemId;
			std::shared_ptr<RasterNoFitPolygon> nfp;
			int posX, posY;
			int weight;
			bool enabled;
			bool operator==(const TotalOverlapMapEntry& hs) const { return std::tie(nfp, posX, posY) == std::tie(hs.nfp, hs.posX, hs.posY); }
		};

		void processVoronoiEntry(TotalOverlapMapEntry &currentEntry);
		void resetVoronoiEntriesCache();

		bool emptyCache;
		int currentCount, toRemoveCount;
		QVector<TotalOverlapMapEntry> currentEntries, toRemoveEntries, toAddEntries;
		const int totalNumItems;
	};
}

#endif // TOTALOVERLAPMAPCACHE_H
