#ifndef TOTALOVERLAPMAP_H
#define TOTALOVERLAPMAP_H

#include "rasternofitpolygon.h"
#include "rasterstrippackingparameters.h"
#include <QDebug>

namespace RASTERVORONOIPACKING {

    class TotalOverlapMap
    {
    public:
        TotalOverlapMap(std::shared_ptr<RasterNoFitPolygon> ifp);
        TotalOverlapMap(int width, int height);
		TotalOverlapMap(QRect &boundingBox);
		virtual ~TotalOverlapMap() { delete[] data; }

        void init(uint _width, uint _height);
        void reset();

		void setWidth(int _width) {
			if (_width > this->width) {
				// Expanding the map buffer
				delete[] data;
				init(_width, this->height);
			}
			this->width = _width;
		}

        void shrink(int pixels) {
            Q_ASSERT_X(width-pixels > 0, "TotalOverlapMap::shrink", "Item does not fit the container");
            this->width = width-pixels;
			if (pixels < 0) {
				//qWarning() << "Expansion over limit not yet implemented!"; return;
				// FIXME: Expand GPU innerift polygon buffer!
				delete[] data;
				init(this->width, this->height);
				return;
			}
            std::fill(data, data+width*height, (float)0.0);
        }
        // Does not expand more than the initial container!
        void expand(int pixels) {
            //#ifdef QT_DEBUG
            //    Q_ASSERT_X(width-pixels <= initialWidth, "TotalOverlapMap::expand", "Container larger than the inital lenght");
            //#endif
            shrink(-pixels);
        }

        void setReferencePoint(QPoint _ref) {reference = _ref;}
        QPoint getReferencePoint() {return reference;}
        int getWidth() {return width;}
        int getHeight() {return height;}
		QRect getRect() { return QRect(-reference, QSize(width, height)); }
        float getValue(const QPoint &pt) {return getLocalValue(pt.x()+reference.x(),pt.y()+reference.y());}
        void setValue(const QPoint &pt, float value) {setLocalValue(pt.x()+reference.x(), pt.y()+reference.y(), value);}

        void addVoronoi(std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos);
        void addVoronoi(std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, float weight);

		virtual QPoint getMinimum(float &minVal, PositionChoice placementHeuristic = BOTTOMLEFT_POS);

        #ifndef CONSOLE
            QImage getImage(); // For debug purposes
        #endif

		void setData(float *_data) { delete[] data; data = _data; }

    protected:
        float *data;
        int width;
        int height;

    private:
        float *scanLine(int y);
        float getLocalValue(int i, int j) {return data[j*width+i];}
        void setLocalValue(int i, int j, float value) {data[j*width+i] = value;}
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
