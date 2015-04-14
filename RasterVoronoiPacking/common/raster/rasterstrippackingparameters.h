#ifndef RASTERSTRIPPACKINGPARAMETERS_H
#define RASTERSTRIPPACKINGPARAMETERS_H

namespace RASTERVORONOIPACKING {
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

		void Copy(RasterStripPackingParameters &source) {
			setNmo(source.getNmo());
			setTimeLimit(source.getTimeLimit());
			setHeuristic(source.getHeuristic());
			setDoubleResolution(source.isDoubleResolution());
			setGpuProcessing(source.isGpuProcessing());
			setCacheMaps(source.isCacheMaps());
			setFixedLength(source.isFixedLength());
		}

	private:
		int Nmo, maxSeconds;
		Heuristic heuristicType;
		bool doubleResolution, gpuProcessing, cacheMaps, fixedLength;
	};
}

#endif //RASTERSTRIPPACKINGPARAMETERS_H