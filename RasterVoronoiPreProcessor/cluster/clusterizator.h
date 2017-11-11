#ifndef CLUSTERIZATOR_H
#define CLUSTERIZATOR_H

#include "../RasterVoronoiPacking/common/packingproblem.h"

namespace CLUSTERING {

	struct Cluster {
		Cluster(std::shared_ptr<RASTERPACKING::RasterNoFitPolygon> _noFitPolygon, QPointF _orbitingPos, qreal _clusterValue, qreal _convexHullArea) : noFitPolygon(_noFitPolygon), orbitingPos(_orbitingPos), clusterValue(_clusterValue), convexHullArea(_convexHullArea), multiplicity(1) {}
		std::shared_ptr<RASTERPACKING::RasterNoFitPolygon> noFitPolygon;
		QPointF orbitingPos;
		qreal clusterValue;
		qreal convexHullArea;
		int multiplicity;
	};

	class Clusterizator {
	public:
		Clusterizator(RASTERPACKING::PackingProblem *_problem);
		~Clusterizator() {}
		QList<Cluster> getAllValidClusters();
		QList<Cluster> getBestClusters(QList<int> rankings);
		QList<Cluster> getBestClusters(int numClusters);
		void getClusteredProblem(RASTERPACKING::PackingProblem &problem, QList<Cluster> &clusters);
		QString getClusteredPuzzle(QString original, QList<Cluster> &clusters, QList<QString> &removedPieces, qreal scaleFixFactor);
		QString getClusterInfo(RASTERPACKING::PackingProblem &clusterProblem, QList<Cluster> &clusters, QString outputXMLName, QList<QString> &removedPieces);

	private:
		void getNextPairNfpList(QList<std::shared_ptr<RASTERPACKING::RasterNoFitPolygon>> &noFitPolygons, QList<std::shared_ptr<RASTERPACKING::RasterNoFitPolygon>>::iterator &curIt);
		std::shared_ptr<RASTERPACKING::RasterNoFitPolygon> getMaximumPairCluster(QList<std::shared_ptr<RASTERPACKING::RasterNoFitPolygon>> noFitPolygons, QPointF &displacement, qreal &value, qreal &area);
		qreal getMaximumClusterPosition(std::shared_ptr<RASTERPACKING::RasterNoFitPolygon> noFitPolygon, QPointF &displacement, qreal &area);
		QList<QPointF> getContourPlacements(std::shared_ptr<RASTERPACKING::RasterNoFitPolygon> noFitPolygon);
		void printCluster(QString fileName, std::shared_ptr<RASTERPACKING::RasterNoFitPolygon> noFitPolygon, QList<QPointF> &displacements);
		qreal getClusterFunction(RASTERPACKING::Polygon &polygon1, RASTERPACKING::Polygon &polygon2, qreal &convexHullArea);
		int checkValidClustering(QList<Cluster> &minClusters, Cluster &candidateCluster);
		bool Clusterizator::checkValidClustering(QList<Cluster> &clusterList);
		bool checkValidClustering(Cluster &candidateCluster);
		void insertNewCluster(QList<Cluster> &minClusters, Cluster &candidateCluster, int numClusters);
		RASTERPACKING::PackingProblem *problem;
		qreal containerWidth, containerHeight;
		qreal weight_compression, weight_intersection, weight_width;

		class SimplifiedPuzzleProblem {
		public:
			SimplifiedPuzzleProblem(QString fileName);
			void save(QString fileName);
			void getPuzzleStream(QTextStream &stream);
			QStringList container;
			QList<unsigned int> multiplicities;
			QList<QStringList> items;
			QList<QStringList> angles;
		};
	};
}

#endif // CLUSTERIZATOR_H