#include "totaloverlapmapcache.h"

using namespace RASTERVORONOIPACKING;

void CachedTotalOverlapMap::processVoronoiEntry(TotalOverlapMapEntry &currentEntry) {
	int itemId = currentEntry.itemId;
	if (!emptyCache && currentEntry == currentEntries[itemId]) {
		if (currentEntry.weight != currentEntries[itemId].weight) toAddEntries << TotalOverlapMapEntry(currentEntry, currentEntry.weight - currentEntries[itemId].weight);
		toRemoveEntries[itemId].enabled = false; toRemoveCount--;
	}
	else toAddEntries << currentEntry;
	currentEntries[itemId] = currentEntry; currentCount++;
}

void CachedTotalOverlapMap::resetVoronoiEntriesCache() {
	currentCount = 0;
	toRemoveEntries = currentEntries; toRemoveCount = totalNumItems - 1;
	toAddEntries.clear();
	emptyCache = false;
}

void CachedTotalOverlapMap::addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, int weight) {
	// Process cache
	processVoronoiEntry(TotalOverlapMapEntry(itemId, nfp, pos, weight));

	// All cache entries were processed, create overlap map
	if (currentCount == totalNumItems - 1) {
		if (!emptyCache && toRemoveCount + toAddEntries.size() < totalNumItems - 1) {
			for (auto entry : toRemoveEntries) if (entry.enabled) TotalOverlapMap::addVoronoi(entry.itemId, entry.nfp, QPoint(entry.posX, entry.posY), -entry.weight);
			for (auto entry : toAddEntries) TotalOverlapMap::addVoronoi(entry.itemId, entry.nfp, QPoint(entry.posX, entry.posY), entry.weight);
		}
		else {
			TotalOverlapMap::reset();
			for (auto entry : currentEntries) if (entry.enabled) TotalOverlapMap::addVoronoi(entry.itemId, entry.nfp, QPoint(entry.posX, entry.posY), entry.weight);
		}
		resetVoronoiEntriesCache();
	}
}

void CachedTotalOverlapMap::addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, int weight, int zoomFactorInt) {
	// Process cache
	processVoronoiEntry(TotalOverlapMapEntry(itemId, nfp, pos, weight));

	// All cache entries were processed, create overlap map
	if (currentCount == totalNumItems - 1) {
		if (!emptyCache && toRemoveCount + toAddEntries.size() < totalNumItems - 1) {
			for (auto entry : toRemoveEntries) if (entry.enabled) TotalOverlapMap::addVoronoi(entry.itemId, entry.nfp, QPoint(entry.posX, entry.posY), -entry.weight, zoomFactorInt);
			for (auto entry : toAddEntries) TotalOverlapMap::addVoronoi(entry.itemId, entry.nfp, QPoint(entry.posX, entry.posY), entry.weight, zoomFactorInt);
		}
		else {
			TotalOverlapMap::reset();
			for (auto entry : currentEntries) if (entry.enabled) TotalOverlapMap::addVoronoi(entry.itemId, entry.nfp, QPoint(entry.posX, entry.posY), entry.weight, zoomFactorInt);
		}
		resetVoronoiEntriesCache();
	}
}

void CachedTotalOverlapMap::addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos) {
	// Process cache
	processVoronoiEntry(TotalOverlapMapEntry(itemId, nfp, pos, 1));

	// All cache entries were processed, create overlap map
	if (currentCount == totalNumItems - 1) {
		if (!emptyCache && toRemoveCount + toAddEntries.size() < totalNumItems - 1) {
			for (auto entry : toRemoveEntries) if (entry.enabled) TotalOverlapMap::addVoronoi(entry.itemId, entry.nfp, QPoint(entry.posX, entry.posY), -entry.weight);
			for (auto entry : toAddEntries) TotalOverlapMap::addVoronoi(entry.itemId, entry.nfp, QPoint(entry.posX, entry.posY), entry.weight);
		}
		else {
			TotalOverlapMap::reset();
			for (auto entry : currentEntries) if (entry.enabled) TotalOverlapMap::addVoronoi(entry.itemId, entry.nfp, QPoint(entry.posX, entry.posY), entry.weight);
		}
		resetVoronoiEntriesCache();
	}
}

void CachedTotalOverlapMap::init(uint _width, uint _height) {
	TotalOverlapMap::init(_width, _height);
	currentCount = 0;
	toAddEntries.clear();
	emptyCache = true;
}