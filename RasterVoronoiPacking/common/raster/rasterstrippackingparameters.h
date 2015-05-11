#ifndef RASTERSTRIPPACKINGPARAMETERS_H
#define RASTERSTRIPPACKINGPARAMETERS_H

namespace RASTERVORONOIPACKING {
	enum ConstructivePlacement { KEPPSOLUTION, RANDOMFIXED, BOTTOMLEFT};
	enum Heuristic { NONE, GLS };

	class RasterStripPackingParameters
	{
	public:
		RasterStripPackingParameters() :
			Nmo(200), maxSeconds(600), heuristicType(GLS), doubleResolution(false),
			gpuProcessing(false), cacheMaps(false), fixedLength(false)
		{} // Default parameters

		void setNmo(int _Nmo) { this->Nmo = _Nmo; }
		int getNmo() { return this->Nmo; }

		void setTimeLimit(int _maxSeconds) { this->maxSeconds = _maxSeconds; }
		int getTimeLimit() { return this->maxSeconds; }

		void setHeuristic(Heuristic _heuristicType) { this->heuristicType = _heuristicType; }
		Heuristic getHeuristic() { return this->heuristicType; }

		void setDoubleResolution(bool val) { this->doubleResolution = val; }
		bool isDoubleResolution() { return this->doubleResolution; }

		void setGpuProcessing(bool val) { this->gpuProcessing = val; }
		bool isGpuProcessing() { return this->gpuProcessing; }

		void setCacheMaps(bool val) { this->cacheMaps = val; }
		bool isCacheMaps() { return this->cacheMaps; }

		void setFixedLength(bool val) { this->fixedLength = val; }
		bool isFixedLength() { return this->fixedLength; }

		void setInitialSolMethod(ConstructivePlacement _initialSolMethod) { this->initialSolMethod = _initialSolMethod; };
		ConstructivePlacement getInitialSolMethod() { return this->initialSolMethod; }

		void settInitialLenght(qreal _initialLenght) { this->initialLenght = _initialLenght; setInitialSolMethod(RANDOMFIXED); } // FIXME: Should the initial solution method be set automatically?
		qreal getInitialLenght() { return this->initialLenght; }

		void Copy(RasterStripPackingParameters &source) {
			setNmo(source.getNmo());
			setTimeLimit(source.getTimeLimit());
			setHeuristic(source.getHeuristic());
			setDoubleResolution(source.isDoubleResolution());
			setGpuProcessing(source.isGpuProcessing());
			setCacheMaps(source.isCacheMaps());
			setFixedLength(source.isFixedLength());
			setInitialSolMethod(source.getInitialSolMethod());
			if (getInitialSolMethod() == RANDOMFIXED) settInitialLenght(source.getInitialLenght());
		}

	private:
		int Nmo, maxSeconds;
		Heuristic heuristicType;
		ConstructivePlacement initialSolMethod;
		qreal initialLenght; // Only used with RANDOMFIXED initial solution
		bool doubleResolution, gpuProcessing, cacheMaps, fixedLength;
	};
}

#endif //RASTERSTRIPPACKINGPARAMETERS_H