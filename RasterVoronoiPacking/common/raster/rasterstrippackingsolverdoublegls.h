#ifndef RASTERSTRIPPACKINGSOLVERDOUBLEGLS_H
#define RASTERSTRIPPACKINGSOLVERDOUBLEGLS_H
#include <memory>
#include "rasterpackingproblem.h"
#include "rasterpackingsolution.h"
#include "rasterstrippackingsolvergls.h"
#include "rasterstrippackingparameters.h"
#include "totaloverlapmap.h"
#include "glsweightset.h"

class MainWindow;

namespace RASTERVORONOIPACKING {

	class RasterStripPackingSolverDoubleGLS : public RasterStripPackingSolverGLS
	{
		friend class MainWindow;
	public:
		RasterStripPackingSolverDoubleGLS(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<RasterPackingProblem> _searchProblem) : RasterStripPackingSolverGLS(_searchProblem) {
			this->originalProblem = _problem;
			this->searchProblem = _searchProblem;
			currentWidth = this->originalProblem->getContainerWidth();
			initialWidth = currentWidth;
			determineMinimumOriginalIfpWidth(_problem);
		}

	private:
		void updateMapsLength(int pixelWidth, RasterStripPackingParameters &params);
		std::shared_ptr<TotalOverlapMap> getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution, RasterStripPackingParameters &params);
		std::shared_ptr<TotalOverlapMap> getTotalOverlapSearchMap(int itemId, int orientation, RasterPackingSolution &solution, RasterStripPackingParameters &params);
		std::shared_ptr<TotalOverlapMap> getRectTotalOverlapMap(int itemId, int orientation, QPoint pos, int width, int height, RasterPackingSolution &solution);
		QPoint getMinimumOverlapSeachPosition(int itemId, int orientation, RasterPackingSolution &solution, RasterStripPackingParameters &params);
		std::shared_ptr<RasterPackingProblem> searchProblem;
	};

}

#endif // RASTERSTRIPPACKINGSOLVERDOUBLEGLS_H
