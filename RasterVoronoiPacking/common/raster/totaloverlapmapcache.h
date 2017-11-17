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

		void init(uint _width, uint _height); // FIXME: Is it really necessary to reimplement?
		void reset() {}
		void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos);
		void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, int weight);
		void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, int weight, int zoomFactorInt);
	protected:
		struct TotalOverlapMapEntry {
			TotalOverlapMapEntry() : isVisible(false) {}
			TotalOverlapMapEntry(std::shared_ptr<RasterNoFitPolygon> _nfp, QPoint pos, int _weight) : nfp(_nfp), posX(pos.x()), posY(pos.y()), weight(_weight), isVisible(false) {}
			TotalOverlapMapEntry(TotalOverlapMapEntry &other, int _weight) : nfp(other.nfp), posX(other.posX), posY(other.posY), weight(_weight), isVisible(false) {}
			std::shared_ptr<RasterNoFitPolygon> nfp;
			int posX, posY;
			int weight;
			bool isVisible;
		};

		virtual void processVoronoiEntry(int itemId, TotalOverlapMapEntry &currentEntry);
		void resetVoronoiEntriesCache();

		int currentCount;
		QList<TotalOverlapMapEntry> toAddEntries, toRemoveEntries;
		QMap<int, TotalOverlapMapEntry> currentEntries;
		const int totalNumItems;
	};

	class CachedTotalRectOverlapMap : public CachedTotalOverlapMap {
	public:
		CachedTotalRectOverlapMap(std::shared_ptr<RasterNoFitPolygon> ifp, int _totalNumItems) : CachedTotalOverlapMap(ifp, _totalNumItems) { };
		CachedTotalRectOverlapMap(int width, int height, QPoint _reference, int _totalNumItems) : CachedTotalOverlapMap(width, height, _reference, _totalNumItems) { };
		~CachedTotalRectOverlapMap() {}

		void setRectangle(QRect &boundingBox);
		void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, int weight);

	private:
		void processVoronoiEntry(int itemId, TotalOverlapMapEntry &currentEntry);
		void checkVisibilitty(TotalOverlapMapEntry &entry);
		int visibleItems;
	};
}

#endif // TOTALOVERLAPMAPCACHE_H
