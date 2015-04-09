#ifndef TOTALOVERLAPMAP_H
#define TOTALOVERLAPMAP_H

#include "rasternofitpolygon.h"
#include <QDebug>

namespace RASTERVORONOIPACKING {

	class CachePlacementInfo {
	public:
		CachePlacementInfo() : oldPosition(QPoint(0, 0)), oldOrientation(0), oldWeight(1.0), placementChanged(false) {}
		~CachePlacementInfo() {}

		void cacheOldPlacement(QPoint oldPosition, int oldOrientation) {
			if (placementChanged) return; // Keep old cached placement
			this->oldPosition = oldPosition;
			this->oldOrientation = oldOrientation;
			placementChanged = true;
		}
		void cacheOldPlacement(QPoint oldPosition, int oldOrientation, qreal oldWeight) {
			if (placementChanged) return; // Keep old cached placement
			this->oldPosition = oldPosition;
			this->oldOrientation = oldOrientation;
			this->oldWeight = oldWeight;
			placementChanged = true;
		}

		QPoint getPosition() const { return this->oldPosition; }
		int getOrientation() const { return this->oldOrientation; }
		qreal getWeight() const { return this->oldWeight; }

		bool changedPlacement() { return placementChanged; }
		void setPlacementChange(bool isChanged) { placementChanged = isChanged; }

	private:
		QPoint oldPosition;
		int oldOrientation;
		qreal oldWeight;
		bool placementChanged;
	};

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
            std::fill(data, data+width*height, (float)0.0);
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
        float getValue(const QPoint &pt) {return getLocalValue(pt.x()+reference.x(),pt.y()+reference.y());}
        void setValue(const QPoint &pt, float value) {setLocalValue(pt.x()+reference.x(), pt.y()+reference.y(), value);}

        void addVoronoi(std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos);
        void addVoronoi(std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, float weight);

        virtual QPoint getMinimum(float &minVal);

        #ifndef CONSOLE
            QImage getImage(); // For debug purposes
        #endif

		void setData(float *_data) { data = _data; }

		// TEST: Cache operations
		void initCacheInfo(int nItems);
		void resetCacheInfo(bool changedValue = false);
		int getCacheCount();
		std::shared_ptr<CachePlacementInfo> getCacheInfo(int orbitingPieceId) { return cacheInfo[orbitingPieceId]; }

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

		QVector<std::shared_ptr<CachePlacementInfo>> cacheInfo;
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
