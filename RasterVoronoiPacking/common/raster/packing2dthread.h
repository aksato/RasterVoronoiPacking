#ifndef PACKING2DTHREAD_H
#define PACKING2DTHREAD_H

#include "packingthread.h"

class Packing2DThread : public PackingThread {
	Q_OBJECT
public:
	Packing2DThread(QObject *parent = 0) { method = RASTERVORONOIPACKING::RANDOM_ENCLOSED; };
	~Packing2DThread() {};

	void setSolver(std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver2D> _solver) {
		solver2d = _solver;
		threadSolution = RASTERVORONOIPACKING::RasterPackingSolution(solver2d->getNumItems());
	}
	void setMethod(RASTERVORONOIPACKING::EnclosedMethod _method) { method = _method; }
	
protected:
	void run();

private:
	void runSquare();
	void runRectangle();
	void costShrinkContainerDimensions(int &curLenght, int &curHeight, RASTERVORONOIPACKING::RasterPackingSolution &currentSolution, const qreal ratio);
	void randomChangeContainerDimensions(int &curLenght, int &curHeight, const qreal ratio);
	bool getShrinkedDimension(int dim, int &newDim, bool length);
	bool checkShrinkSizeConstraint(int &curLength, int &curHeight, int &reducedLength, int &reducedHeight, qreal ratio);

	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver2D> solver2d;
	RASTERVORONOIPACKING::EnclosedMethod method;
};

#endif // PACKING2DTHREAD_H
