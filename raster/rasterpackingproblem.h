#ifndef RASTERPACKINGPROBLEM_H
#define RASTERPACKINGPROBLEM_H
#include <memory>
#include <QVector>
#include "rasternofitpolygon.h"
#include "rasterinnerfitpolygon.h"
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

    private:
        unsigned int id;
        unsigned int pieceType;
        unsigned int angleCount;
    };

    class RasterPackingProblem
    {
    public:
        RasterPackingProblem();
        RasterPackingProblem(RASTERPREPROCESSING::PackingProblem &problem);
        ~RasterPackingProblem() {}

    public:
        bool load(RASTERPREPROCESSING::PackingProblem &problem);
        std::shared_ptr<RasterPackingItem> getItem(int id) {return items[id];}

        std::shared_ptr<RasterNoFitPolygonSet> getIfps() {return innerFitPolygons;}
        std::shared_ptr<RasterNoFitPolygonSet> getNfps() {return noFitPolygons;}
        int count() {return items.size();}
        int getItemType(int id) {return items[id]->getPieceType();}
        int getContainerWidth() {return containerWidth;}
        qreal getScale() {return scale;}

    private:
//        QPair<int,int> getIdsFromRasterPreProblem(QString polygonName, int angleValue, QHash<QString, int> &pieceIndexMap, QHash<QPair<int,int>, int> &pieceAngleMap);

        int containerWidth;
        unsigned int maxOrientations;
        QVector<std::shared_ptr<RasterPackingItem>> items;
        std::shared_ptr<RasterNoFitPolygonSet> noFitPolygons;
        std::shared_ptr<RasterNoFitPolygonSet> innerFitPolygons;
        qreal scale;
    };
}
#endif // RASTERPACKINGPROBLEM_H
