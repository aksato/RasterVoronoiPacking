#ifndef RASTERPACKINGSOLUTION_H
#define RASTERPACKINGSOLUTION_H

#include <QVector>
#include <QPoint>
#include <QDebug>

namespace RASTERVORONOIPACKING {

    class RasterItemPlacement {
    public:
        RasterItemPlacement()  : pos(QPoint(0,0)) , orientation(0) {}
        ~RasterItemPlacement() {}

        void setPos(QPoint newPos) {this->pos = newPos;}
        void setOrientation(int newOrientation) {this->orientation = newOrientation;}
        QPoint getPos() const {return this->pos;}
        int getOrientation() const {return this->orientation;}

    private:
        QPoint pos;
        int orientation;
    };

    class RasterPackingSolution
    {
    public:
        RasterPackingSolution();
        RasterPackingSolution(int numItems);

        int getNumItems() const {return placements.size();}
        void setPosition(int id, QPoint newPos) {placements[id].setPos(newPos);}
        QPoint getPosition(int id) const {return placements[id].getPos();}
        void setOrientation(int id, int newOrientation) {placements[id].setOrientation(newOrientation);}
        int getOrientation(int id) const {return placements[id].getOrientation();}

    private:
        QVector<RasterItemPlacement> placements;
    };


}
QDebug operator<<(QDebug dbg, const RASTERVORONOIPACKING::RasterPackingSolution &c);

#endif // RASTERPACKINGSOLUTION_H
