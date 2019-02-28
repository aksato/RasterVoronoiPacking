#ifndef TOTALOVERLAPMAP_H
#define TOTALOVERLAPMAP_H

#include "rasternofitpolygon.h"
#include "rasterstrippackingparameters.h"
#include <Eigen/Core>
#ifdef GRAYSCALE
#include<iostream>
#endif

namespace RASTERVORONOIPACKING {

    class TotalOverlapMap
    {
    public:
        TotalOverlapMap(std::shared_ptr<RasterNoFitPolygon> ifp, int _cuttingStockLength = -1);
		TotalOverlapMap(int width, int height, QPoint _reference, int _cuttingStockLength = -1);
		TotalOverlapMap(QRect &boundingBox, int _cuttingStockLength = -1);
		virtual ~TotalOverlapMap() { if(data != nullptr) delete[] data; }

        void init(uint _width, uint _height);
        virtual void reset();

		virtual void setDimensions(int _width, int _height) {
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
		int getCuttingStockLength() { return this->cuttingStockLength; }

        virtual void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos);
		virtual void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, int weight);
		virtual void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, int weight, int zoomFactorInt);
		virtual void changeTotalItems(int _totalNumItems) {} // FIXME: Better way to evaluate partial cached overlap map
		void addToMatrix(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, Eigen::Matrix< unsigned int, Eigen::Dynamic, Eigen::Dynamic > &overlapMatrix);
		void addToMatrixCuda(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, std::shared_ptr<quint32> d_overlapMatrix);
		void setDataFromMatrix(Eigen::Matrix< unsigned int, Eigen::Dynamic, Eigen::Dynamic > &overlapMatrix, Eigen::Matrix< unsigned int, Eigen::Dynamic, 1 > &weightVec);
		quint32 getMinimum(QPoint &minPt);
		quint32 getMinimum(QPoint &minPt, int &stockLocation);
		quint32 getBottomLeft(QPoint &minPt, bool borderOk = true);
		void setZoomFactor(int _zoomFactor);
        QImage getImage(); // For debug purposes
		QImage getZoomImage(int _width, int _height, QPoint &displacement); // For debug purposes
		void maskCuttingStock(); // For debug purposes

		void setData(quint32 *_data) { delete[] data; data = _data; }
		quint32 *getData() { return data; }
		#ifdef GRAYSCALE
		void print() { for (int i = 0; i < width*height; i++) std::cout << data[i] << std::endl; }
		#endif

    protected:
        quint32 *data;
        int width;
        int height;
		const int originalWidth, originalHeight;
		QPoint reference;
		//#ifdef QT_DEBUG
		int initialWidth;
		int cuttingStockLength;
		int zoomFactor;
		//#endif

    private:
		quint32 *scanLine(int x);
		quint32 getLocalValue(int i, int j) { return data[j*width + i]; }
		void setLocalValue(int i, int j, quint32 value) { data[j*width + i] = value; }
        bool getLimits(QPoint relativeOrigin, int vmWidth, int vmHeight, QRect &intersection);
		bool getLimits(QPoint relativeOrigin, int vmWidth, int vmHeight, QRect &intersection, int zoomFactorInt);
		void findMinimum(quint32 &minVal, int &minid, int &curid, int blockSize);
    };

    class TotalOverlapMapSet
    {
    public:
		TotalOverlapMapSet(int numItems);
        TotalOverlapMapSet(int numberOfOrientations, int numItems);

        void addOverlapMap(int orbitingPieceId, int orbitingAngleId, std::shared_ptr<TotalOverlapMap> ovm);
        std::shared_ptr<TotalOverlapMap> getOverlapMap(int orbitingPieceId, int orbitingAngleId);
		void clear() { mapSet = QVector<std::shared_ptr<TotalOverlapMap>>(mapSet.length()); }

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

	class TotalOverlapMapCuda : public TotalOverlapMap
	{
	public:
		TotalOverlapMapCuda(std::shared_ptr<RasterNoFitPolygon> ifp, int _numItems, int _cuttingStockLength = -1);
		TotalOverlapMapCuda(int width, int height, QPoint _reference, int _numItems, int _cuttingStockLength = -1);
		TotalOverlapMapCuda(QRect &boundingBox, int _numItems, int _cuttingStockLength = -1);
		~TotalOverlapMapCuda();

		void initCuda(uint _width, uint _height);

		void reset();
		void setDimensions(int _width, int _height);
		void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos);
		void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, int weight) {}
		void addVoronoi(int itemId, std::shared_ptr<RasterNoFitPolygon> nfp, QPoint pos, int weight, int zoomFactorInt) {}
		void changeTotalItems(int _totalNumItems) {} // FIXME: Better way to evaluate partial cached overlap map

	private:
		int numItems;
	};
}

#endif // TOTALOVERLAPMAP_H
