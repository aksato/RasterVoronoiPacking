#ifndef PACKINGPROBLEM_H
#define PACKINGPROBLEM_H

#include<memory>
#include<QVector>
#include<QPointF>
#include<QString>
#include<QStringList>
#include<QDebug>
#include<QImage>
#include<QXmlStreamWriter>

namespace POLYBOOLEAN {
	struct PAREA;
}

//typedef QVector<QPointF> PolygonF;

namespace RASTERPACKING {

	class Polygon : public QPolygonF {
    public:
		Polygon() : QPolygonF() {}
		Polygon(QString _name) : QPolygonF() { this->name = _name; }
		Polygon(const QVector<QPointF> &points) : QPolygonF(points) {}
        ~Polygon() {}
        QStringList getXML();
        QString getName() {return this->name;}

		static std::shared_ptr<Polygon> getConvexHull(Polygon& polygon);
		static std::shared_ptr<Polygon> getConvexHull(Polygon& polygon1, Polygon& polygon2);
        static std::shared_ptr<Polygon> getNofitPolygon(std::shared_ptr<Polygon> staticPolygon, std::shared_ptr<Polygon> orbitingPolygon); // FIXME: Does not work for concave polygons
		int *getRasterImage(QPoint &RP, qreal scale, int &width, int &height);
		void getRasterBoundingBox(QPoint &RP, qreal scale, int &width, int &height);
		void fromPolybool(POLYBOOLEAN::PAREA *area, qreal scale);
		qreal getArea();
		void setBoundingBoxMinX(qreal _minX) { this->minX = _minX; }
		void setBoundingBoxMaxX(qreal _maxX) { this->maxX = _maxX; }
		void setBoundingBoxMinY(qreal _minY) { this->minY = _minY; }
		void setBoundingBoxMaxY(qreal _maxY) { this->maxY = _maxY; }
		void getBoundingBox(qreal &_minX, qreal &_maxX, qreal &_minY, qreal &_maxY) {
			_minX = this->minX; _maxX = this->maxX; _minY = this->minY; _maxY = this->maxY;
		}

    private:
        QVector<QPair<QPointF,QPointF>> degLines;
        QVector<QPointF> degNodes;
        QString name;
		qreal minX, maxX, minY, maxY;
    };

    class PackingComponent {
    public:
        PackingComponent() {}
        virtual ~PackingComponent() {}

        void setName(QString _name) {this->name = _name;}
        QString getName() {return this->name;}
        void setPolygon(std::shared_ptr<Polygon> polygon) {this->polygon = polygon;}
        std::shared_ptr<Polygon> getPolygon() {return this->polygon;}
        virtual QStringList getXML() = 0;

    protected:
        QString name;
        std::shared_ptr<Polygon> polygon;
    };

    class Container : public PackingComponent {
    public:
        Container() {}
        Container(QString _name, int _multiplicity);
        ~Container() {}

        void setMultiplicity(unsigned int _multiplicity) {this->multiplicity = _multiplicity;}
        unsigned int getMultiplicity() {return this->multiplicity;}
        QStringList getXML();

    private:
        unsigned int multiplicity;
    };

    class Piece : public PackingComponent {
    public:
        Piece() {}
        Piece(QString _name, int _multiplicity);
        ~Piece() {}

        void setMultiplicity(unsigned int _multiplicity) {this->multiplicity = _multiplicity;}
        unsigned int getMultiplicity() {return this->multiplicity;}
        void addOrientation(unsigned int angle) {this->orientations.push_back(angle);}
        int getOrientationsCount() {return this->orientations.size();}
        QStringList getXML();
		int decomposeConvex();

        // Iterators
        QVector<unsigned int>::iterator orbegin() {return this->orientations.begin();}
        QVector<unsigned int>::iterator orend() {return this->orientations.end();}
        QVector<unsigned int>::const_iterator corbegin() {return this->orientations.cbegin();}
        QVector<unsigned int>::const_iterator corend() {return this->orientations.cend();}
		QVector<std::shared_ptr<Polygon>>::iterator cConvexbegin() { return this->convexPartitions.begin(); }
		QVector<std::shared_ptr<Polygon>>::iterator cConvexend() { return this->convexPartitions.end(); }

    private:
        unsigned int multiplicity;
        QVector<unsigned int> orientations;
		QVector<std::shared_ptr<Polygon>> convexPartitions;
    };

    class GeometricTool : public PackingComponent {
    public:
        GeometricTool() {}
        ~GeometricTool() {}

        void setStaticName(QString name) {this->staticName = name;}
        QString getStaticName() {return this->staticName;}
        void setOrbitingName(QString name) {this->orbitingName = name;}
        QString getOrbitingName() {return this->orbitingName;}
        void setStaticAngle(int angle) {this->staticAngle = angle;}
        int getStaticAngle() {return this->staticAngle;}
        void setOrbitingAngle(int angle) {this->orbitingAngle = angle;}
        int getOrbitingAngle() {return this->orbitingAngle;}
        virtual QStringList getXML() = 0;

    protected:
        QString staticName, orbitingName;
        int staticAngle, orbitingAngle;
    };

    class NoFitPolygon : public GeometricTool {
    public:
        NoFitPolygon() {}
        ~NoFitPolygon() {}
        QStringList getXML();
    };

    class InnerFitPolygon : public GeometricTool {
    public:
        InnerFitPolygon() {}
        ~InnerFitPolygon() {}
        QStringList getXML();
    };

    class RasterGeometricTool : public GeometricTool {
    public:
		RasterGeometricTool() {} // FIXME: Remove it
        RasterGeometricTool(int _width, int _height) : width(_width), height(_height) {}
        ~RasterGeometricTool() {}

        QString getFileName() {return this->fileName;}
        QPoint getReferencePoint() {return this->referencePoint;}
        qreal getScale() {return this->scale;}
		int getWidth() { return this->width; }
		int getHeight() { return this->height; }
        virtual QStringList getXML() = 0;
        void setFileName(QString fname) {this->fileName = fname;}
        void setReferencePoint(QPoint RP) {this->referencePoint = RP;}
        void setScale(qreal scale) {this->scale = scale;}
		void setWidth(int width) {this->width = width;}
		void setHeight(int height) { this->height = height; }

    protected:
        QString fileName;
        QPoint referencePoint;
        qreal scale;
		int width, height;
    };

    class RasterNoFitPolygon : public RasterGeometricTool {
    public:
        RasterNoFitPolygon() {
            polygon = nullptr;
        }
		RasterNoFitPolygon(std::shared_ptr<NoFitPolygon> nfp, int _width, int _height) : RasterGeometricTool(_width, _height) {
            polygon = nullptr;
            this->staticName  = nfp->getStaticName();
            this->staticAngle = nfp->getStaticAngle();
            this->orbitingName  = nfp->getOrbitingName();
            this->orbitingAngle = nfp->getOrbitingAngle();
        }
        ~RasterNoFitPolygon() {}
        QStringList getXML();
    };

    class RasterInnerFitPolygon : public RasterGeometricTool {
    public:
        RasterInnerFitPolygon() {
            polygon = nullptr;
        }
		RasterInnerFitPolygon(std::shared_ptr<InnerFitPolygon> ifp, int _width, int _height) : RasterGeometricTool(_width, _height) {
            polygon = nullptr;
            this->staticName  = ifp->getStaticName();
            this->staticAngle = ifp->getStaticAngle();
            this->orbitingName  = ifp->getOrbitingName();
            this->orbitingAngle = ifp->getOrbitingAngle();
        }
        ~RasterInnerFitPolygon() {}
        QStringList getXML();

    };

	namespace CLUSTERING {
		struct ClusterPiece {
			ClusterPiece(QString _pieceName, int _angle, QPointF _offset) : pieceName(_pieceName), angle(_angle), offset(_offset) {}
			QString pieceName;
			int angle;
			QPointF offset;
		};
		typedef QList<ClusterPiece> Cluster;
	}

	enum Symmetry {NONE, PAIR};
    class PackingProblem {
    public:
		PackingProblem() : nfpDataSymmetry(NONE) {}
        ~PackingProblem() {}

        bool load(QString fileName);
		bool load(QString fileName, QString fileType, qreal scale = 1.0, qreal auxScale = 1.0);
		bool loadCFREFP(QTextStream &stream, qreal scale, qreal auxScale = 1.0);
		bool loadTerashima(QTextStream &stream, int specificContainer = -1);
		bool loadClusterInfo(QString fileName);
        bool save(QString fileName, QString binFileName = "", QString clusterInfo = "");
		bool savePuzzle(QString fileName);
		qreal getTotalItemsArea();
		QString getFolder() { return this->folder; }

        void setName(QString _name) {this->name = _name;}
        QString getName() {return this->name;}
        void setAuthor(QString _author) {this->author = _author;}
        QString getAuthor() {return this->author;}
        void setDate(QString _date) {this->date = _date;}
        QString getDate() {return this->date;}
        void setDescription(QString _description) {this->description = _description;}
        QString getDescription() {return this->description;}
        bool copyHeader(QString fileName);
		Symmetry getDataSymmetry() { return this->nfpDataSymmetry; }
        QByteArray *getNfpDataRef() { return &this->nfpData; }
        quint32* loadBinaryNofitPolygons(QVector<QPair<quint32, quint32>>& sizes, QVector<QPoint>& rps);

		void resizeRasterNoFitPolygon() { this->rasterNofitPolygons.resize(nofitPolygons.size()); }
        void addRasterNofitPolygon(std::shared_ptr<RasterNoFitPolygon> rasterNfp) {this->rasterNofitPolygons.push_back(rasterNfp);}
		void addRasterNofitPolygon(std::shared_ptr<RasterNoFitPolygon> rasterNfp, int index) { this->rasterNofitPolygons[index] = rasterNfp; }
        void addRasterInnerfitPolygon(std::shared_ptr<RasterInnerFitPolygon> rasterIfp) {this->rasterInnerfitPolygons.push_back(rasterIfp);}

		int getItemsCount() { return std::accumulate(pieces.begin(), pieces.end(), 0, [](int lhs, std::shared_ptr<Piece> rhs){return lhs + rhs->getMultiplicity(); }); }
        int getContainersCount() {return containers.size();}
        int getPiecesCount() {return pieces.size();}
        int getNofitPolygonsCount() {return nofitPolygons.size();}
        int getInnerfitPolygonsCount() {return innerfitPolygons.size();}

        void addContainer(std::shared_ptr<Container> _container) {this->containers.push_back(_container);}
        void addPiece(std::shared_ptr<Piece> _piece) {this->pieces.push_back(_piece);}
        void addNoFitPolygon(std::shared_ptr<NoFitPolygon> _nofitpolygon) {this->nofitPolygons.push_back(_nofitpolygon);}
        void addInnerFitPolygon(std::shared_ptr<InnerFitPolygon> _innerfitpolygon) {this->innerfitPolygons.push_back(_innerfitpolygon);}

        // Iterators
        QList<std::shared_ptr<Container>>::iterator cbegin() {return this->containers.begin();}
        QList<std::shared_ptr<Container>>::iterator cend() {return this->containers.end();}
        QList<std::shared_ptr<Container>>::const_iterator ccbegin() {return this->containers.cbegin();}
        QList<std::shared_ptr<Container>>::const_iterator ccend() {return this->containers.cend();}

        QList<std::shared_ptr<Piece>>::iterator pbegin() {return this->pieces.begin();}
        QList<std::shared_ptr<Piece>>::iterator pend() {return this->pieces.end();}
        QList<std::shared_ptr<Piece>>::const_iterator cpbegin() {return this->pieces.cbegin();}
        QList<std::shared_ptr<Piece>>::const_iterator cpend() {return this->pieces.cend();}
		QList<std::shared_ptr<Piece>>::iterator erasep(QList<std::shared_ptr<Piece>>::iterator pos) {return this->pieces.erase(pos);}

		std::shared_ptr<NoFitPolygon> getNofitPolygon(int index) { return this->nofitPolygons[index]; }
		QVector<std::shared_ptr<NoFitPolygon>>::iterator nfpbegin() { return this->nofitPolygons.begin(); }
		QVector<std::shared_ptr<NoFitPolygon>>::iterator nfpend() { return this->nofitPolygons.end(); }
		QVector<std::shared_ptr<NoFitPolygon>>::const_iterator cnfpbegin() { return this->nofitPolygons.cbegin(); }
		QVector<std::shared_ptr<NoFitPolygon>>::const_iterator cnfpend() { return this->nofitPolygons.cend(); }
		QVector<std::shared_ptr<NoFitPolygon>>::iterator erasenfp(QVector<std::shared_ptr<NoFitPolygon>>::iterator pos) { return this->nofitPolygons.erase(pos); }

        QList<std::shared_ptr<InnerFitPolygon>>::iterator ifpbegin() {return this->innerfitPolygons.begin();}
        QList<std::shared_ptr<InnerFitPolygon>>::iterator ifpend() {return this->innerfitPolygons.end();}
        QList<std::shared_ptr<InnerFitPolygon>>::const_iterator cifpbegin() {return this->innerfitPolygons.cbegin();}
        QList<std::shared_ptr<InnerFitPolygon>>::const_iterator cifpend() {return this->innerfitPolygons.cend();}

		QVector<std::shared_ptr<RasterNoFitPolygon>>::iterator rnfpbegin() { return this->rasterNofitPolygons.begin(); }
		QVector<std::shared_ptr<RasterNoFitPolygon>>::iterator rnfpend() { return this->rasterNofitPolygons.end(); }
		QVector<std::shared_ptr<RasterNoFitPolygon>>::const_iterator crnfpbegin() { return this->rasterNofitPolygons.cbegin(); }
		QVector<std::shared_ptr<RasterNoFitPolygon>>::const_iterator crnfpend() { return this->rasterNofitPolygons.cend(); }

        QList<std::shared_ptr<RasterInnerFitPolygon>>::iterator rifpbegin() {return this->rasterInnerfitPolygons.begin();}
        QList<std::shared_ptr<RasterInnerFitPolygon>>::iterator rifpend() {return this->rasterInnerfitPolygons.end();}
        QList<std::shared_ptr<RasterInnerFitPolygon>>::const_iterator crifpbegin() {return this->rasterInnerfitPolygons.cbegin();}
        QList<std::shared_ptr<RasterInnerFitPolygon>>::const_iterator crifpend() {return this->rasterInnerfitPolygons.cend();}

		QMap<QString, CLUSTERING::Cluster>::iterator clusterbegin() { return this->clusteredPieces.begin(); }
		QMap<QString, CLUSTERING::Cluster>::iterator clusterend() { return this->clusteredPieces.end(); }
		QMap<QString, CLUSTERING::Cluster>::const_iterator cclusterbegin() { return this->clusteredPieces.cbegin(); }
		QMap<QString, CLUSTERING::Cluster>::const_iterator cclusterend() { return this->clusteredPieces.cend(); }
		CLUSTERING::Cluster getCluster(QString name) { return this->clusteredPieces[name]; }
		QString getOriginalProblem() { return this->originalProblem; }

		qreal getMaxLength() { return this->maxLength; }
		qreal getMaxWidth() { return this->maxWidth; }
    private:
		bool loadCFREFP(QString &fileName, qreal scale, qreal auxScale = 1.0);
		bool saveClusterInfo(QXmlStreamWriter &stream, QString clusterInfo);

        QString name, author, date, description, folder;

        QList<std::shared_ptr<Container>> containers;
        QList<std::shared_ptr<Piece>> pieces;
		QVector<std::shared_ptr<NoFitPolygon>> nofitPolygons;
        QList<std::shared_ptr<InnerFitPolygon>> innerfitPolygons;
        QVector<std::shared_ptr<RasterNoFitPolygon>> rasterNofitPolygons;
        QList<std::shared_ptr<RasterInnerFitPolygon>> rasterInnerfitPolygons;
		QMap<QString, CLUSTERING::Cluster> clusteredPieces;
		QString originalProblem;
		qreal maxLength, maxWidth;
		QString nfpDataFileName;
		Symmetry nfpDataSymmetry;
        QByteArray nfpData;
    };
}

#endif // PACKINGPROBLEM_H
