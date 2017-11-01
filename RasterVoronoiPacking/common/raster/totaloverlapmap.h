#ifndef TOTALOVERLAPMAP_H
#define TOTALOVERLAPMAP_H

#include "rasternofitpolygon.h"
#include "rasterstrippackingparameters.h"

namespace RASTERVORONOIPACKING {

    class TotalOverlapMap
    {
    public:
        TotalOverlapMap(std::shared_ptr<RasterNoFitPolygon> ifp);
		TotalOverlapMap(int width, int height, QPoint _reference);
		TotalOverlapMap(QRect &boundingBox);
		virtual ~TotalOverlapMap() { delete[] data; }

        virtual void init(uint _width, uint _height);
        virtual void reset();

		void setDimensions(int _width, int _height) {
			Q_ASSERT_X(_width > 0 && _height > 0, "TotalOverlapMap::shrink", "Item does not fit the container");
			if (_width > this->width || _height > this->height) {
				// Expanding the map buffer
				delete[] data;
				init(_width, _height);
			}
			this->width = _width; this->height = _height;
		}
		void setWidth(int _width) { setDimensions(_width, this->height); }
		void setRelativeWidth(int deltaPixels) { setWidth(this->originalWidth - deltaPixels); }
		void setRelativeDimensions(int deltaPixelsX, int deltaPixelsY) { setDimensions(this->originalWidth - deltaPixelsX, this->originalHeight - deltaPixelsY); }

        void setReferencePoint(QPoint _ref) {reference = _ref;}
        QPoint getReferencePoint() {return reference;}
        int getWidth() {return width;}
        int getHeight() {return height;}
		const int getOriginalWidth() {return originalWidth;}
		const int getOriginalHeight() { return originalHeight; }
		QRect getRect() { return QRect(-reference, QSize(width, height)); }
		quint32 getValue(const QPoint &pt) { return getLocalValue(pt.x() + reference.x(), pt.y() + reference.y()); }
		void setValue(const QPoint &pt, quint32 value) { setLocalValue(pt.x() + reference.x(), pt.y() + reference.y(), value); }

        virtual void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos);
		virtual void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, int weight);
		virtual void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, int weight, int zoomFactorInt);
		quint32 getMinimum(QPoint &minPt);

        #ifndef CONSOLE
            QImage getImage(); // For debug purposes
			QImage getZoomImage(int _width, int _height, QPoint &displacement); // For debug purposes
        #endif

		void setData(quint32 *_data) { delete[] data; data = _data; }

    protected:
        quint32 *data;
        int width;
        int height;
		const int originalWidth, originalHeight;
		#ifdef QT_DEBUG
		int initialWidth;
		#endif

    private:
		quint32 *scanLine(int y);
		quint32 getLocalValue(int i, int j) { return data[j*width + i]; }
		void setLocalValue(int i, int j, quint32 value) { data[j*width + i] = value; }
        bool getLimits(QPoint relativeOrigin, int vmWidth, int vmHeight, QRect &intersection);
		bool getLimits(QPoint relativeOrigin, int vmWidth, int vmHeight, QRect &intersection, int zoomFactorInt);
        QPoint reference;
    };

    class TotalOverlapMapSet
    {
    public:
		TotalOverlapMapSet(int numItems);
        TotalOverlapMapSet(int numberOfOrientations, int numItems);

        void addOverlapMap(int orbitingPieceId, int orbitingAngleId, std::shared_ptr<TotalOverlapMap> ovm);
        std::shared_ptr<TotalOverlapMap> getOverlapMap(int orbitingPieceId, int orbitingAngleId);
		void clear() { mapSet.clear(); }

        //QHash<int, std::shared_ptr<TotalOverlapMap>>::const_iterator cbegin() {return mapSet.cbegin();}
        //QHash<int, std::shared_ptr<TotalOverlapMap>>::const_iterator cend() {return mapSet.cend();}

		void setShrinkVal(int val) { this->shrinkValX = val; }
		void setShrinkVal(int valX, int valY) { this->shrinkValX = valX; this->shrinkValY = valY; }
		int getShrinkValX() { return this->shrinkValX; }
		int getShrinkValY() { return this->shrinkValY; }
    private:
        //QHash<int, std::shared_ptr<TotalOverlapMap>> mapSet;
		void initializeSet(int numItems);
		QVector<std::shared_ptr<TotalOverlapMap>> mapSet;
		int numAngles, shrinkValX, shrinkValY;
    };

}

#endif // TOTALOVERLAPMAP_H
