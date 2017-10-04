#ifndef RASTERSTRIPPACKINGPARAMETERS_H
#define RASTERSTRIPPACKINGPARAMETERS_H

namespace RASTERVORONOIPACKING {
	enum ConstructivePlacement { KEEPSOLUTION, RANDOMFIXED, BOTTOMLEFT};
	enum Heuristic { NONE, GLS };
	enum PositionChoice { BOTTOMLEFT_POS, RANDOM_POS, LIMITS_POS, CONTOUR_POS};
	enum EnclosedMethod { SQUARE, RANDOM_ENCLOSED, BAGPIPE };

	class RasterStripPackingParameters
	{
	public:
		RasterStripPackingParameters() :
			Nmo(200), maxSeconds(600), heuristicType(GLS), doubleResolution(false),
			fixedLength(false), maxIterations(0), rectangularPacking(false)
		{} // Default parameters

		void setNmo(int _Nmo) { this->Nmo = _Nmo; }
		int getNmo() { return this->Nmo; }

		void setTimeLimit(int _maxSeconds) { this->maxSeconds = _maxSeconds; }
		int getTimeLimit() { return this->maxSeconds; }

		void setIterationsLimit(int _maxIterations) { this->maxIterations = _maxIterations; }
		int getIterationsLimit() { return this->maxIterations; }

		void setHeuristic(Heuristic _heuristicType) { this->heuristicType = _heuristicType; }
		Heuristic getHeuristic() { return this->heuristicType; }

		void setDoubleResolution(bool val) { this->doubleResolution = val; }
		bool isDoubleResolution() { return this->doubleResolution; }

		void setFixedLength(bool val) { this->fixedLength = val; }
		bool isFixedLength() { return this->fixedLength; }

		void setInitialSolMethod(ConstructivePlacement _initialSolMethod) { this->initialSolMethod = _initialSolMethod; };
		ConstructivePlacement getInitialSolMethod() { return this->initialSolMethod; }

		void setInitialLenght(qreal _initialLenght) { this->initialLenght = _initialLenght; setInitialSolMethod(RANDOMFIXED); } // FIXME: Should the initial solution method be set automatically?
		qreal getInitialLenght() { return this->initialLenght; }

		void setPlacementCriteria(PositionChoice _placementCriteria) { this->placementCriteria = _placementCriteria; }
		PositionChoice getPlacementCriteria() { return this->placementCriteria; }

		void setClusterFactor(qreal _clusterFactor) { this->clusterFactor = _clusterFactor; }
		qreal getClusterFactor() { return this->clusterFactor; }
		
		void setRectangularPacking(bool val) { this->rectangularPacking = val; }
		bool isRectangularPacking() { return this->rectangularPacking; }

		void setRectangularPackingMethod(EnclosedMethod method) { this->rectangularPackingMethod = method; }
		EnclosedMethod getRectangularPackingMethod() { return this->rectangularPackingMethod; }

		void Copy(RasterStripPackingParameters &source) {
			setNmo(source.getNmo());
			setTimeLimit(source.getTimeLimit());
			setIterationsLimit(source.getIterationsLimit());
			setHeuristic(source.getHeuristic());
			setDoubleResolution(source.isDoubleResolution());
			setFixedLength(source.isFixedLength());
			setInitialSolMethod(source.getInitialSolMethod());
			if (getInitialSolMethod() == RANDOMFIXED) setInitialLenght(source.getInitialLenght());
			setPlacementCriteria(source.getPlacementCriteria());
			setClusterFactor(source.getClusterFactor());
			setRectangularPacking(source.isRectangularPacking());
			setRectangularPackingMethod(source.getRectangularPackingMethod());
		}

	private:
		int Nmo, maxSeconds, maxIterations;
		Heuristic heuristicType;
		ConstructivePlacement initialSolMethod;
		PositionChoice placementCriteria; // FIXME: Debug
		qreal initialLenght; // Only used with RANDOMFIXED initial solution
		bool doubleResolution, fixedLength, rectangularPacking;
		EnclosedMethod rectangularPackingMethod;
		qreal clusterFactor;
	};
}

#endif //RASTERSTRIPPACKINGPARAMETERS_H