#ifndef RASTERNOFITPOLYGON_H
#define RASTERNOFITPOLYGON_H
#include "memory"
#include <QHash>
#include <QImage>
#include <QPoint>
#include <vector>
#include <qDebug>

namespace RASTERPACKING {class PackingProblem;}

namespace RASTERVORONOIPACKING {
    class RasterNoFitPolygon {
    public:
		RasterNoFitPolygon(QImage _image, QPoint _origin) : origin(_origin) { setMatrix(_image); }
		RasterNoFitPolygon(quint32 *_matrix, int _width, int _height, QPoint _origin) : origin(_origin), matrix(_matrix), w(_width), h(_height) {}
		RasterNoFitPolygon(int _width, int _height, QPoint _origin) : origin(_origin), w(_width), h(_height) { matrix = new quint32[w * h](); }
		~RasterNoFitPolygon() { if (!matrix) { delete[] matrix; matrix = nullptr; } } // FIXME: delete or not? depends on the constructor, which is bad

        void setOrigin(QPoint origin) {this->origin = origin;}
        QPoint getOrigin() {return this->origin;}
        int getOriginX() {return this->origin.x();}
        int getOriginY() {return this->origin.y();}

        void setMatrix(QImage image);
		quint32 getPixel(int i, int j) { return matrix[j*w + i]; }
		int width() {return w;}
		int height() {return h;}
		QRect boundingBox() { return QRect(-this->origin, QSize(w, h)); }

		quint32 *getMatrix() { return matrix; }
		void setMatrix(quint32 *_matrix) { matrix = _matrix; }

    private:
        QPoint origin;
		quint32 *matrix;
		int w, h;
    };

    class RasterNoFitPolygonSet
    {
    public:
        RasterNoFitPolygonSet();
        RasterNoFitPolygonSet(int numberOfOrientations);

        bool load(RASTERPACKING::PackingProblem &problem);
        void addRasterNoFitPolygon(int staticPieceTypeId, int staticAngleId, int orbitingPieceTypeId, int orbitingAngleId, std::shared_ptr<RasterNoFitPolygon> nfp);
        std::shared_ptr<RasterNoFitPolygon> getRasterNoFitPolygon(int staticPieceTypeId, int staticAngleId, int orbitingPieceTypeId, int orbitingAngleId);
        void eraseRasterNoFitPolygon(int staticPieceTypeId, int staticAngleId, int orbitingPieceTypeId, int orbitingAngleId);
        void clear();

    private:
        QHash<QPair<int,int>, std::shared_ptr<RasterNoFitPolygon>> Nfps;
        int numAngles;
    };
}
#endif // RASTERNOFITPOLYGON_H
