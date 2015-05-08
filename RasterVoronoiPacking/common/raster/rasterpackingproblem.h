#ifndef RASTERPACKINGPROBLEM_H
#define RASTERPACKINGPROBLEM_H
#include <memory>
#include <QVector>
#include "rasternofitpolygon.h"
#include "rasterinnerfitpolygon.h"
#include <QDebug>
class QString;
namespace RASTERPREPROCESSING {class PackingProblem; class RasterNoFitPolygon;}

namespace RASTERVORONOIPACKING {
    class RasterPackingItem {
    public:
        RasterPackingItem() {}
        RasterPackingItem(unsigned int _id, unsigned int pType, unsigned int aCount) : id(_id), pieceType(pType), angleCount(aCount) {}
        ~RasterPackingItem() {}

        void setId(unsigned int _id) {this->id = _id;}
        unsigned int getId() {return this->id;}
        void setPieceType(unsigned int pType) {this->pieceType = pType;}
        unsigned int getPieceType() {return this->pieceType;}
        void setAngleCount(unsigned int aCount) {this->angleCount = aCount;}
        unsigned int getAngleCount() {return this->angleCount;}

        void setPieceName(QString pName) {this->pieceName = pName;}
        QString getPieceName() {return this->pieceName;}
        void addAngleValue(int angle) {this->angleValues.push_back(angle);}
        int getAngleValue(int id) {return this->angleValues.at(id);}

		void setBoundingBox(int _minX, int _maxX, int _minY, int _maxY) {
			this->minX = _minX; this->maxX = _maxX; this->minY = _minY; this->maxY = _maxY;
		}
		void getBoundingBox(int &_minX, int &_maxX, int &_minY, int &_maxY) {
			_minX = this->minX; _maxX = this->maxX; _minY = this->minY; _maxY = this->maxY;
		}

    private:
        unsigned int id;
        unsigned int pieceType;
        unsigned int angleCount;
		int minX, maxX, minY, maxY;

        QString pieceName;
        QVector<int> angleValues;
    };

    class RasterPackingProblem
    {
    public:
        RasterPackingProblem();
        RasterPackingProblem(RASTERPREPROCESSING::PackingProblem &problem);
        ~RasterPackingProblem() {}

    public:
        bool load(RASTERPREPROCESSING::PackingProblem &problem, bool loadGPU = false);
        std::shared_ptr<RasterPackingItem> getItem(int id) {return items[id];}

        std::shared_ptr<RasterNoFitPolygonSet> getIfps() {return innerFitPolygons;}
        std::shared_ptr<RasterNoFitPolygonSet> getNfps() {return noFitPolygons;}
        int count() {return items.size();}
        int getItemType(int id) {return items[id]->getPieceType();}
        int getContainerWidth() {return containerWidth;}
        QString getContainerName() {return containerName;}
        qreal getScale() {return scale;}

		static void getProblemGPUMemRequirements(RASTERPREPROCESSING::PackingProblem &problem, size_t &ifpTotalMem, size_t &ifpMaxMem, size_t &nfpTotalMem);

    private:
//        QPair<int,int> getIdsFromRasterPreProblem(QString polygonName, int angleValue, QHash<QString, int> &pieceIndexMap, QHash<QPair<int,int>, int> &pieceAngleMap);

        int containerWidth;
        QString containerName;
        unsigned int maxOrientations;
        QVector<std::shared_ptr<RasterPackingItem>> items;
        std::shared_ptr<RasterNoFitPolygonSet> noFitPolygons;
        std::shared_ptr<RasterNoFitPolygonSet> innerFitPolygons;
        qreal scale;
    };
}
#endif // RASTERPACKINGPROBLEM_H
