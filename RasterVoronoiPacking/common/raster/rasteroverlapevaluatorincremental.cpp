#include "rasteroverlapevaluatorincremental.h"

using namespace RASTERVORONOIPACKING;

RasterTotalOverlapMapEvaluatorIncremental::RasterTotalOverlapMapEvaluatorIncremental(std::shared_ptr<RasterPackingProblem> _problem, bool cuttingStock) : 
	RasterTotalOverlapMapEvaluatorGLS(_problem, false, cuttingStock), 
	currentSolution(_problem->count()) {
	initializeMaps(_problem);
}

RasterTotalOverlapMapEvaluatorIncremental::RasterTotalOverlapMapEvaluatorIncremental(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<GlsWeightSet> _glsWeights, bool cuttingStock) : 
	RasterTotalOverlapMapEvaluatorGLS(_problem, _glsWeights, false, cuttingStock), 
	currentSolution(_problem->count())  {
	initializeMaps(_problem);
}

void RasterTotalOverlapMapEvaluatorIncremental::initializeMaps(std::shared_ptr<RasterPackingProblem> _problem) {
	for (int i = 0; i < _problem->count(); i++)
		for (uint curAngle = 0; curAngle < _problem->getItem(i)->getAngleCount(); curAngle++)
			RasterTotalOverlapMapEvaluatorGLS::getTotalOverlapMap(i, curAngle, currentSolution);
}

// Determines the item total overlap map for a given orientation in a solution
std::shared_ptr<TotalOverlapMap> RasterTotalOverlapMapEvaluatorIncremental::getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution& solution) {
	std::shared_ptr<TotalOverlapMap> currrentPieceMap = maps.getOverlapMap(itemId, orientation);
	return currrentPieceMap;
}

void RasterTotalOverlapMapEvaluatorIncremental::updateWeights(RasterPackingSolution& solution, QVector<quint32>& overlaps, quint32 maxOverlap) {
	QVector<quint32> increments;
	std::transform(overlaps.begin(), overlaps.end(), std::back_inserter(increments), [&maxOverlap](quint32 v) { return qRound(100.0 * (qreal)v / (qreal)maxOverlap); });

	// Update all overlap maps with increments
	int overlapIndex = 0;
	for (int itemId = 0; itemId < problem->count(); itemId++) {
		int i = 0;
		for (; i < itemId + 1; i++) overlapIndex++;
		for (; i < problem->count(); i++) {
			// Update NFP for item i induced by item itemId
			for (uint curAngle = 0; curAngle < problem->getItem(i)->getAngleCount(); curAngle++)
				maps.getOverlapMap(i, curAngle)->addVoronoi(i, problem->getNfps()->getRasterNoFitPolygon(
					problem->getItemType(itemId), currentSolution.getOrientation(itemId), problem->getItemType(i), curAngle),
					currentSolution.getPosition(itemId), increments[overlapIndex]);

			// Update NFP for item itemId induced by item i
			for (uint curAngle = 0; curAngle < problem->getItem(itemId)->getAngleCount(); curAngle++)
				maps.getOverlapMap(itemId, curAngle)->addVoronoi(itemId, problem->getNfps()->getRasterNoFitPolygon(
					problem->getItemType(i), currentSolution.getOrientation(i), problem->getItemType(itemId), curAngle),
					currentSolution.getPosition(i), increments[overlapIndex]);

			overlapIndex++;
		}
	}

	RasterTotalOverlapMapEvaluatorGLS::updateWeights(solution, overlaps, maxOverlap);
}

void RasterTotalOverlapMapEvaluatorIncremental::signalNewItemPosition(int itemId, int orientation, QPoint newPos) {
	if (newPos.x() == currentSolution.getPosition(itemId).x() && newPos.y() == currentSolution.getPosition(itemId).y() && orientation == currentSolution.getOrientation(itemId))
		return;

	// Update all other overlap maps given the new static position
	for (int i = 0; i < problem->count(); i++) {
		if (i == itemId) continue;
		for (uint curAngle = 0; curAngle < problem->getItem(i)->getAngleCount(); curAngle++) {
			std::shared_ptr<TotalOverlapMap> currrentPieceMap = maps.getOverlapMap(i, curAngle);

			// Remove nofit polygon from old position. TODO: different weight?
			currrentPieceMap->addVoronoi(i, problem->getNfps()->getRasterNoFitPolygon(
				problem->getItemType(itemId), currentSolution.getOrientation(itemId), problem->getItemType(i), curAngle), 
				currentSolution.getPosition(itemId), -getWeight(i, itemId)); 

			// Add nofit polygon to new position
			currrentPieceMap->addVoronoi(i, problem->getNfps()->getRasterNoFitPolygon(
				problem->getItemType(itemId), orientation, problem->getItemType(i), curAngle), 
				newPos, getWeight(i, itemId));
		}
	}

	// Update postion
	currentSolution.setPosition(itemId, newPos);
	currentSolution.setOrientation(itemId, orientation);
}