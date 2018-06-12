#ifndef RASTERSTRIPPACKINGPARAMETERS_H
#define RASTERSTRIPPACKINGPARAMETERS_H

#define DEFAULT_RDEC 0.04
#define DEFAULT_RINC 0.01

namespace RASTERVORONOIPACKING {
	enum ConstructivePlacement { KEEPSOLUTION, RANDOMFIXED, BOTTOMLEFT };
	enum Heuristic { NONE, GLS };
	enum CompactionMode { STRIPPACKING, SQUAREPACKING, RECTRNDPACKING, RECTBAGPIPEPACKING, CUTTINGSTOCK};

	class RasterStripPackingParameters
	{
	public:
		RasterStripPackingParameters() :
			Nmo(200), maxSeconds(600), heuristicType(GLS), zoomFactor(1), initialLength(-1), initialHeight(-1), compaction(STRIPPACKING), cacheMaps(false),
			fixedLength(false), maxIterations(0), rdec(DEFAULT_RDEC), rinc(DEFAULT_RINC)
		{} // Default parameters

		RasterStripPackingParameters(Heuristic _heuristicType, int _zoomFactor) :
			Nmo(200), maxSeconds(600), heuristicType(_heuristicType), zoomFactor(_zoomFactor), initialLength(-1), initialHeight(-1), compaction(STRIPPACKING), cacheMaps(false),
			fixedLength(false), maxIterations(0), rdec(DEFAULT_RDEC), rinc(DEFAULT_RINC)
		{} // Default parameters with specific solver parameters

		void setNmo(int _Nmo) { this->Nmo = _Nmo; }
		int getNmo() { return this->Nmo; }

		void setTimeLimit(int _maxSeconds) { this->maxSeconds = _maxSeconds; }
		int getTimeLimit() { return this->maxSeconds; }

		void setIterationsLimit(int _maxIterations) { this->maxIterations = _maxIterations; }
		int getIterationsLimit() { return this->maxIterations; }

		void setHeuristic(Heuristic _heuristicType) { this->heuristicType = _heuristicType; }
		Heuristic getHeuristic() { return this->heuristicType; }

		void setFixedLength(bool val) { this->fixedLength = val; }
		bool isFixedLength() { return this->fixedLength; }

		void setInitialSolMethod(ConstructivePlacement _initialSolMethod) { this->initialSolMethod = _initialSolMethod; };
		ConstructivePlacement getInitialSolMethod() { return this->initialSolMethod; }

		void setZoomFactor(int _zoomFactor) { this->zoomFactor = _zoomFactor; }
		int getZoomFactor() { return this->zoomFactor; }

		void setResizeChangeRatios(qreal _ratioDecrease, qreal _ratioIncrease) { this->rdec = _ratioDecrease; this->rinc = _ratioIncrease; }
		qreal getRdec() { return this->rdec; }
		qreal getRinc() { return this->rinc; }

		void setCompaction(CompactionMode _compaction) { this->compaction = _compaction; }
		CompactionMode getCompaction() { return this->compaction; }

		void setInitialDimensions(int _initialLength, int _initialHeight = -1) { initialLength = _initialLength; initialHeight = _initialHeight; }
		int getInitialLength() { return this->initialLength; }
		int getInitialHeight() { return this->initialHeight; }

		bool isCacheMaps() { return this->cacheMaps; }
		void setCacheMaps(bool _cacheMaps) { this->cacheMaps = _cacheMaps; }

		void Copy(RasterStripPackingParameters &source) {
			setNmo(source.getNmo());
			setTimeLimit(source.getTimeLimit());
			setIterationsLimit(source.getIterationsLimit());
			setHeuristic(source.getHeuristic());
			setFixedLength(source.isFixedLength());
			setInitialSolMethod(source.getInitialSolMethod());
			setZoomFactor(source.getZoomFactor());
			setResizeChangeRatios(source.getRdec(), source.getRinc());
			setCompaction(source.getCompaction());
			setInitialDimensions(source.getInitialLength(), source.getInitialHeight());
			setCacheMaps(source.isCacheMaps());
		}

	private:
		int Nmo, maxSeconds, maxIterations;
		Heuristic heuristicType;
		ConstructivePlacement initialSolMethod;
		CompactionMode compaction;
		int initialLength, initialHeight;
		bool fixedLength;
		int zoomFactor;
		qreal rdec, rinc;
		bool cacheMaps;
	};
}

#endif //RASTERSTRIPPACKINGPARAMETERS_H