#ifndef PACKING2DTHREAD_H
#define PACKING2DTHREAD_H

#include "packingthread.h"

class Packing2DThread : public PackingThread {
	Q_OBJECT
public:
	Packing2DThread(QObject *parent = 0) : bagpipeDirection(1) {};
	~Packing2DThread() {};
	
protected:
	void run();

private:
	void runSquare();
	void runRectangle();
	void randomChangeContainerDimensions(int &curLength, int &curHeight, qreal &curRealLength, qreal &curRealHeight, const qreal ratio);
	void costChangeContainerDimensions(int &curLength, int &curHeight, qreal &curRealLength, qreal &curRealHeight, RASTERVORONOIPACKING::RasterPackingSolution &currentSolution, const qreal ratio);
	void bagpipeChangeContainerDimensions(int &curLength, int &curHeight, qreal &curRealLength, qreal &curRealHeight, int bestArea, const qreal ratio);
	int bagpipeDirection;
};

#endif // PACKING2DTHREAD_H
