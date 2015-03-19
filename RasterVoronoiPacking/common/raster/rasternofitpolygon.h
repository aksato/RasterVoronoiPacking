#ifndef RASTERNOFITPOLYGON_H
#define RASTERNOFITPOLYGON_H
#include "memory"
#include <QHash>
#include <QImage>
#include <QPoint>
#include <vector>

namespace RASTERPACKING {class PackingProblem;}

namespace RASTERVORONOIPACKING {
    class RasterNoFitPolygon {
    public:
//        RasterNoFitPolygon(QImage _image, QPoint _origin, qreal _maxD) : origin(_origin), image(_image) , maxD(_maxD) {}
        RasterNoFitPolygon(QImage _image, QPoint _origin, qreal _maxD) : origin(_origin), maxD(_maxD) {setMatrix(_image);}
        ~RasterNoFitPolygon() {}

        void setOrigin(QPoint origin) {this->origin = origin;}
//        void setImage(QImage image) {this->image = image;}
        QPoint getOrigin() {return this->origin;}
        int getOriginX() {return this->origin.x();}
        int getOriginY() {return this->origin.y();}
//        QImage getImage() {return this->image;}
        qreal getMaxD() {return this->maxD;}

        void setMatrix(QImage image);
        quint8 getPixel(int i, int j) {return matrix[j][i];}
        int width() {return (int)matrix[0].size();}
        int height() {return (int)matrix.size();}

		void save(QString fname); // For debug purposes

    private:
        QPoint origin;
//        QImage image;
        qreal maxD;
        std::vector< std::vector<quint8> > matrix;
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
