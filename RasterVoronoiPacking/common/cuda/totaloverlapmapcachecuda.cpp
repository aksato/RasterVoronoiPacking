#include "totaloverlapmapcachecuda.h"

using namespace RASTERVORONOIPACKING;

void CachedTotalOverlapMapCuda::processVoronoiEntry(int itemId, TotalOverlapMapEntry &currentEntry) {
	currentCount++;
	TotalOverlapMapEntry &cachedEntry = currentEntries[itemId];
	//if (!currentEntries.empty() && currentEntry == cachedEntry) {
	if (!currentEntries.empty() && currentEntry.nfp == cachedEntry.nfp && currentEntry.posX == cachedEntry.posX && currentEntry.posY == cachedEntry.posY) {
		if (currentEntry.weight != cachedEntry.weight) {
			toAddEntries << TotalOverlapMapEntry(currentEntry, currentEntry.weight - cachedEntry.weight);
			cachedEntry.weight = currentEntry.weight;
		}
	}
	else {
		toAddEntries << currentEntry;
		toRemoveEntries << cachedEntry;
		cachedEntry = currentEntry;
	}
}

void CachedTotalOverlapMapCuda::resetVoronoiEntriesCache() {
	currentCount = 0;
	toRemoveEntries.clear();
	toAddEntries.clear();
}

void CachedTotalOverlapMapCuda::addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, int weight) {
	// Process cache
	processVoronoiEntry(itemId, TotalOverlapMapEntry(nfp, pos, weight));

	// All cache entries were processed, create overlap map
	if (currentCount == totalNumItems - 1) {
		if (toRemoveEntries.size() + toAddEntries.size() < totalNumItems - 1) {
			for (auto entry : toRemoveEntries) TotalOverlapMapCuda::addVoronoi(0, entry.nfp, QPoint(entry.posX, entry.posY), -entry.weight);
			for (auto entry : toAddEntries) TotalOverlapMapCuda::addVoronoi(0, entry.nfp, QPoint(entry.posX, entry.posY), entry.weight);
		}
		else {
			TotalOverlapMapCuda::reset();
			for (auto entry : currentEntries) TotalOverlapMapCuda::addVoronoi(0, entry.nfp, QPoint(entry.posX, entry.posY), entry.weight);
		}
		resetVoronoiEntriesCache();
	}
}

void CachedTotalOverlapMapCuda::addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, int weight, int zoomFactorInt) {
	// Process cache
	processVoronoiEntry(itemId, TotalOverlapMapEntry(nfp, pos, weight));

	// All cache entries were processed, create overlap map
	if (currentCount == totalNumItems - 1) {
		if (toRemoveEntries.size() + toAddEntries.size() < totalNumItems - 1) {
			for (auto entry : toRemoveEntries) TotalOverlapMapCuda::addVoronoi(0, entry.nfp, QPoint(entry.posX, entry.posY), -entry.weight, zoomFactorInt);
			for (auto entry : toAddEntries) TotalOverlapMapCuda::addVoronoi(0, entry.nfp, QPoint(entry.posX, entry.posY), entry.weight, zoomFactorInt);
		}
		else {
			TotalOverlapMapCuda::reset();
			for (auto entry : currentEntries) TotalOverlapMapCuda::addVoronoi(0, entry.nfp, QPoint(entry.posX, entry.posY), entry.weight, zoomFactorInt);
		}
		resetVoronoiEntriesCache();
	}
}

void CachedTotalOverlapMapCuda::addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos) {
	// Process cache
	processVoronoiEntry(itemId, TotalOverlapMapEntry(nfp, pos, 1));

	// All cache entries were processed, create overlap map
	if (currentCount == totalNumItems - 1) {
		if (toRemoveEntries.size() + toAddEntries.size() < totalNumItems - 1) {
			for (auto entry : toRemoveEntries) TotalOverlapMapCuda::addVoronoi(0, entry.nfp, QPoint(entry.posX, entry.posY), -entry.weight);
			for (auto entry : toAddEntries) TotalOverlapMapCuda::addVoronoi(0, entry.nfp, QPoint(entry.posX, entry.posY), entry.weight);
		}
		else {
			TotalOverlapMapCuda::reset();
			for (auto entry : currentEntries) TotalOverlapMapCuda::addVoronoi(0, entry.nfp, QPoint(entry.posX, entry.posY), entry.weight);
		}
		resetVoronoiEntriesCache();
	}
}

void CachedTotalOverlapMapCuda::setDimensions(int _width, int _height) {
	TotalOverlapMapCuda::setDimensions(_width, _height);
	resetVoronoiEntriesCache();
	currentEntries.clear();
}