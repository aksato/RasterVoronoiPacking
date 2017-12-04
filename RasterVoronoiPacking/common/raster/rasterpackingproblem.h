#ifndef RASTERPACKINGPROBLEM_H
#define RASTERPACKINGPROBLEM_H
#include <memory>
#include <QVector>
#include "rasternofitpolygon.h"
#include "rasterinnerfitpolygon.h"
#include "raster/rasterpackingsolution.h"
#include <QDebug>
class QString;
namespace RASTERPACKING {class PackingProblem; class RasterNoFitPolygon;}

class MainWindow;

namespace RASTERVORONOIPACKING {
    class RasterPackingItem {
    public:
		RasterPackingItem(unsigned int _id, unsigned int pType, unsigned int aCount, std::shared_ptr<QPolygonF> pol) : id(_id), pieceType(pType), angleCount(aCount), polygon(pol) {}
        ~RasterPackingItem() {}

        void setId(unsigned int _id) {this->id = _id;}
        unsigned int getId() {return this->id;}
        void setPieceType(unsigned int pType) {this->pieceType = pType;}
        unsigned int getPieceType() {return this->pieceType;}
        void setAngleCount(unsigned int aCount) {this->angleCount = aCount;}
        unsigned int getAngleCount() {return this->angleCount;}
		std::shared_ptr<QPolygonF> getPolygon() { return this->polygon; }

        void setPieceName(QString pName) {this->pieceName = pName;}
        QString getPieceName() {return this->pieceName;}
        void addAngleValue(int angle) {this->angleValues.push_back(angle);}
        int getAngleValue(int id) {return this->angleValues.at(id);}
		int getOrientationFromAngle(int angle) { return this->angleValues.indexOf(angle); }
		void setBoundingBox(qreal _minX, qreal _maxX, qreal _minY, qreal _maxY) {
			this->minX = _minX; this->maxX = _maxX; this->minY = _minY; this->maxY = _maxY;
		}
		void getBoundingBox(qreal &_minX, qreal &_maxX, qreal &_minY, qreal &_maxY) {
			_minX = this->minX; _maxX = this->maxX; _minY = this->minY; _maxY = this->maxY;
		}
		qreal getMaxX(int orientation) {
			if (getAngleValue(orientation) == 0) return this->maxX;
			if (getAngleValue(orientation) == 90) return -this->minY;
			if (getAngleValue(orientation) == 180) return -this->minX;
			if (getAngleValue(orientation) == 270) return this->maxY;
			return 0; // FIXME: Implement continuous rotations?
		}
		qreal getMinX(int orientation) {
			if (getAngleValue(orientation) == 0) return this->minX;
			if (getAngleValue(orientation) == 90) return -this->maxY;
			if (getAngleValue(orientation) == 180) return -this->maxX;
			if (getAngleValue(orientation) == 270) return this->minY;
			return 0; // FIXME: Implement continuous rotations?
		}

		qreal getMaxY(int orientation) {
			if (getAngleValue(orientation) == 0) return this->maxY;
			if (getAngleValue(orientation) == 90) return this->maxX;
			if (getAngleValue(orientation) == 180) return -this->minY;
			if (getAngleValue(orientation) == 270) return -this->minX;
			return 0; // FIXME: Implement continuous rotations?
		}

		qreal getMinY(int orientation) {
			if (getAngleValue(orientation) == 0) return this->minY;
			if (getAngleValue(orientation) == 90) return this->minX;
			if (getAngleValue(orientation) == 180) return -this->maxY;
			if (getAngleValue(orientation) == 270) return -this->maxX;
			return 0; // FIXME: Implement continuous rotations?
		}

    private:
        unsigned int id;
        unsigned int pieceType;
        unsigned int angleCount;
		qreal minX, maxX, minY, maxY;

        QString pieceName;
        QVector<int> angleValues;
		std::shared_ptr<QPolygonF> polygon; // For output purposes
    };

    class RasterPackingProblem
    {
		friend class ::MainWindow;
    public:
        RasterPackingProblem();
        RasterPackingProblem(RASTERPACKING::PackingProblem &problem);
        ~RasterPackingProblem() {}

        virtual bool load(RASTERPACKING::PackingProblem &problem);
        std::shared_ptr<RasterPackingItem> getItem(int id) {return items[id];}
		QVector<std::shared_ptr<RasterPackingItem>>::iterator ibegin() { return items.begin(); }
		QVector<std::shared_ptr<RasterPackingItem>>::iterator iend() { return items.end(); }
		qreal getDensity(RasterPackingSolution &solution);
		qreal getRectangularDensity(RasterPackingSolution &solution);

        std::shared_ptr<RasterNoFitPolygonSet> getIfps() {return innerFitPolygons;}
        std::shared_ptr<RasterNoFitPolygonSet> getNfps() {return noFitPolygons;}
        int count() {return items.size();}
        int getItemType(int id) {return items[id]->getPieceType();}
        int getContainerWidth() {return containerWidth;}
		int getContainerHeight() { return containerHeight; }
		int getMaxWidth() { return this->maxWidth; }
		int getMaxHeight() { return this->maxHeight; }
        QString getContainerName() {return containerName;}
        qreal getScale() {return scale;}

		// Processing of nfp values
		quint32 getDistanceValue(int itemId1, QPoint pos1, int orientation1, int itemId2, QPoint pos2, int orientation2);
		bool areOverlapping(int itemId1, QPoint pos1, int orientation1, int itemId2, QPoint pos2, int orientation2);
		void getIfpBoundingBox(int itemId, int orientation, int &bottomLeftX, int &bottomLeftY, int &topRightX, int &topRightY);

    protected:
        int containerWidth, containerHeight;
		int maxWidth, maxHeight;
        QString containerName;
        unsigned int maxOrientations;
        QVector<std::shared_ptr<RasterPackingItem>> items;
		std::shared_ptr<RasterPackingItem> container;
        std::shared_ptr<RasterNoFitPolygonSet> noFitPolygons;
        std::shared_ptr<RasterNoFitPolygonSet> innerFitPolygons;
        qreal scale;
		qreal totalArea;

	private:
		quint32 *loadBinaryNofitPolygons(QString fileName, QVector<QPair<quint32, quint32>> &sizes, QVector<QPoint> &rps);
		quint32 getNfpValue(int itemId1, QPoint pos1, int orientation1, int itemId2, QPoint pos2, int orientation2);
		// Debug functions. TODO: remove them!
		qreal getCurrentWidth(RasterPackingSolution &solution);
		qreal getCurrentHeight(RasterPackingSolution &solution);
		qreal getOriginalWidth();
		qreal getOriginalHeight();
    };

	struct RasterPackingClusterItem {
		RasterPackingClusterItem(QString _pieceName, int _id, int _angle, QPoint _offset) : pieceName(_pieceName), id(_id), angle(_angle), offset(_offset) {}
		QString pieceName;
		int id;
		int angle;
		QPoint offset;
	};
	typedef QList<RasterPackingClusterItem> RasterPackingCluster;

	class RasterPackingClusterProblem : public RasterPackingProblem
	{
	friend class MainWindow;
	public:
		RasterPackingClusterProblem() : RasterPackingProblem() {};
		RasterPackingClusterProblem(RASTERPACKING::PackingProblem &problem) : RasterPackingProblem(problem) {};
		~RasterPackingClusterProblem() {}

		bool load(RASTERPACKING::PackingProblem &problem);
		std::shared_ptr<RASTERVORONOIPACKING::RasterPackingProblem> getOriginalProblem() { return this->originalProblem; }
		void convertSolution(RASTERVORONOIPACKING::RasterPackingSolution &solution);

	private:
		QMap<int, RasterPackingCluster> clustersMap;
		std::shared_ptr<RASTERVORONOIPACKING::RasterPackingProblem> originalProblem;
	};
}
#endif // RASTERPACKINGPROBLEM_H
