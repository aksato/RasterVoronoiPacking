#ifndef TOTALOVERLAPMAP_H
#define TOTALOVERLAPMAP_H

#include "rasternofitpolygon.h"
#include <QDebug>

namespace RASTERVORONOIPACKING {

    class TotalOverlapMap
    {
    public:
        TotalOverlapMap(std::shared_ptr<RasterNoFitPolygon> ifp);
        TotalOverlapMap(int width, int height);
        virtual ~TotalOverlapMap() {}

        void init(uint _width, uint _height);
        void reset();

        void shrink(int pixels) {
            Q_ASSERT_X(width-pixels > 0, "TotalOverlapMap::shrink", "Item does not fit the container");
            this->width = width-pixels;
            std::fill(data, data+width*height, 0);
        }
        // Does not expand more than the initial container!
        void expand(int pixels) {
            #ifdef QT_DEBUG
                Q_ASSERT_X(width-pixels <= initialWidth, "TotalOverlapMap::expand", "Container larger than the inital lenght");
            #endif
            shrink(-pixels);
        }

        void setReferencePoint(QPoint _ref) {reference = _ref;}
        QPoint getReferencePoint() {return reference;}
        int getWidth() {return width;}
        int getHeight() {return height;}
        qreal getValue(const QPoint &pt) {return getLocalValue(pt.x()+reference.x(),pt.y()+reference.y());}
        void setValue(const QPoint &pt, qreal value) {setLocalValue(pt.x()+reference.x(), pt.y()+reference.y(), value);}

        void addVoronoi(std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos);
        void addVoronoi(std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, qreal weight);

        virtual QPoint getMinimum(qreal &minVal);

        #ifndef CONSOLE
            QImage getImage(); // For debug purposes
			void save(QString fname); // For debug purposes
        #endif

    protected:
        qreal *data;
        int width;
        int height;

    private:
        qreal *scanLine(int y);
        qreal getLocalValue(int i, int j) {return data[j*width+i];}
        void setLocalValue(int i, int j, qreal value) {data[j*width+i] = value;}
        bool getLimits(QPoint relativeOrigin, int vmWidth, int vmHeight, QRect &intersection);

        #ifdef QT_DEBUG
        int initialWidth;
        #endif
        QPoint reference;
    };

    class TotalOverlapMapSet
    {
    public:
        TotalOverlapMapSet();
        TotalOverlapMapSet(int numberOfOrientations);

        void addOverlapMap(int orbitingPieceId, int orbitingAngleId, std::shared_ptr<TotalOverlapMap> ovm);
        std::shared_ptr<TotalOverlapMap> getOverlapMap(int orbitingPieceId, int orbitingAngleId);

        QHash<int, std::shared_ptr<TotalOverlapMap>>::const_iterator cbegin() {return mapSet.cbegin();}
        QHash<int, std::shared_ptr<TotalOverlapMap>>::const_iterator cend() {return mapSet.cend();}

    private:
        QHash<int, std::shared_ptr<TotalOverlapMap>> mapSet;
        int numAngles;
    };

}
#endif // TOTALOVERLAPMAP_H
