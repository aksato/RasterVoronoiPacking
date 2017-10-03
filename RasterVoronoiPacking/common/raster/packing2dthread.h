#ifndef PACKING2DTHREAD_H
#define PACKING2DTHREAD_H

#include "packingthread.h"

class Packing2DThread : public PackingThread {
	Q_OBJECT
public:
	Packing2DThread(QObject *parent = 0) {};
	~Packing2DThread() {};

	void setSolver(std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver2D> _solver) {
		solver2d = _solver;
		threadSolution = RASTERVORONOIPACKING::RasterPackingSolution(solver2d->getNumItems());
	}

protected:
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver2D> solver2d;
	void run();

signals:
	// GUI signals
	void solutionGenerated(const RASTERVORONOIPACKING::RasterPackingSolution &solution, int length, int height);
	void dimensionUpdated(const RASTERVORONOIPACKING::RasterPackingSolution &solution, int length, int height, int totalItNum, qreal elapsed, uint seed);
};

enum EnclosedMethod {RANDOM_ENCLOSED, BAGPIPE};

class PackingEnclosedThread : public Packing2DThread {
	Q_OBJECT
public:
	PackingEnclosedThread(QObject *parent = 0) { method = RANDOM_ENCLOSED; };
	~PackingEnclosedThread() {};

	void setMethod(EnclosedMethod _method) { method = _method; }
protected:
	void run();

private:
	void runRandom();
	void runBagpipe();

	EnclosedMethod method;
};

#endif // PACKING2DTHREAD_H
