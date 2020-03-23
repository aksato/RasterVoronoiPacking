#ifndef RASTERPACKINGCUDAPROBLEM_H
#define RASTERPACKINGCUDAPROBLEM_H
#include "raster/rasterpackingproblem.h"
#include <memory>
#include <QVector>
#include <QString>

namespace RASTERVORONOIPACKING {
	class RasterPackingCudaProblem : public RasterPackingProblem
	{
	public:
		RasterPackingCudaProblem();
		RasterPackingCudaProblem(RASTERPACKING::PackingProblem &problem);
		~RasterPackingCudaProblem() {}

		bool load(RASTERPACKING::PackingProblem &problem);

	private:
		quint32 * loadBinaryNofitPolygonsOnDevice(RASTERPACKING::PackingProblem& problem, QVector<QPair<quint32, quint32>> &sizes, QVector<QPoint> &rps);
	};
}
#endif // RASTERPACKINGCUDAPROBLEM_H
