#include "../RasterVoronoiPacking/common/packingproblem.h"
#include "polydecomp-keil\polygon.h"

using namespace RASTERPACKING;

int Piece::decomposeConvex() {
	this->convexPartitions.clear();
	std::shared_ptr<RASTERPACKING::Polygon> curPol = getPolygon();

	// Create polygon structure to decompose
	POLYDECOMP::Polygon incPoly;
	for (auto it = curPol->begin(); it != curPol->end(); it++) {
		QPointF curPt = *it;
		incPoly.push(POLYDECOMP::Point(curPt.x(), curPt.y()));
	}

	// Decompose polygon
	incPoly.makeCCW();
	POLYDECOMP::EdgeList diags = incPoly.decomp();
	std::vector<POLYDECOMP::Polygon> polys = incPoly.slice(incPoly, diags);

	// Convert to RASTERPACKING structure and add to the piece
	for (auto it = polys.begin(); it != polys.end(); it++) {
		std::shared_ptr<RASTERPACKING::Polygon> partitionPol(new RASTERPACKING::Polygon());
		POLYDECOMP::Polygon curPol = *it;
		for (auto it2 = curPol.begin(); it2 != curPol.end(); it2++) *partitionPol << (QPointF((*it2).x, (*it2).y));
		this->convexPartitions << partitionPol;
	}

	return polys.size() > 1;
}

bool PackingProblem::loadTerashima(QTextStream &stream, int specificContainer) {
	unsigned int polygonid = 0;
	int countCont;
	stream >> countCont;
	int count = 0;
	int includePieceMin, includePieceMax;
	includePieceMin = 0;
	for (int i = 0; i < countCont; i++) {
		int num;
		stream >> num;
		if (i == specificContainer) { includePieceMin = count; includePieceMax = count + num; }
		count += num;
	}
	if(specificContainer == -1) includePieceMax = count;

	// Add container
	int width, height;
	stream >> width >> height;
	{
		std::shared_ptr<Container> curContainer(new Container("board0", 1));
		QString containerName = "polygon" + QString::number(polygonid); polygonid++;
		std::shared_ptr<Polygon> curPolygon(new Polygon(containerName));
		//curPolygon->fromPolybool(container->getShape()->getInnerData(), scale*auxScale);
		*curPolygon << QPointF(0,0) << QPointF(width, 0) << QPointF(width, height) << QPointF(0, height);
		curContainer->setPolygon(curPolygon);
		this->containers.push_back(curContainer);
	}

	// Add pieces
	{
		int curX, curY;
		for (; polygonid < count + 1; polygonid++) {
			int numVertexes;
			stream >> numVertexes;
			std::shared_ptr<Piece> curPiece(new Piece("piece" + QString::number(polygonid-1), 1));
			curPiece->addOrientation(0);
			QString pieceName = "polygon" + QString::number(polygonid);
			std::shared_ptr<Polygon> curPolygon(new Polygon(pieceName));
			for (int i = 0; i < numVertexes; i++) {
				stream >> curX >> curY;
				*curPolygon << QPointF(curX, curY);
			}
			curPiece->setPolygon(curPolygon);
			if(polygonid-1 >= includePieceMin && polygonid - 1 < includePieceMax) this->pieces.push_back(curPiece);
		}
	}

	return true;
}