#include "args/args.hxx"
#include "packingproblem.h"
#include "raster/rasterpackingproblem.h"
#include "cuda/rasterpackingcudaproblem.h"
#include "cuda/rasteroverlapevaluatorcudagls.h"
#include "cuda/glsweightsetcuda.h"
#include "raster/rasteroverlapevaluator.h"
#include "raster/rasteroverlapevaluatorfull.h"
#include "raster/rasteroverlapevaluatormatrixgls.h"
#include "raster/rasterstrippackingparameters.h"
#include "raster/rasterstrippackingsolver.h"
#include "raster/rastersquarepackingcompactor.h"
#include <iostream>
#include <QString>
#include <QDir>
#include <QFileInfo>
#include <QTime>

#define REPETITIONS 1000

using namespace RASTERVORONOIPACKING;

QVector<std::shared_ptr<RasterPackingSolution>> solutions;
int mapcount = 1;

std::shared_ptr<GlsWeightSet> generateRandomWeigths(int count) {
	std::shared_ptr<GlsWeightSet> weights = std::shared_ptr<GlsWeightSet>(new GlsWeightSet(count));
	for (int i = 0; i < count; i++) 
		for (int j = 0; j < count; j++) {
			if (i == j) continue;
			weights->addWeight(i, j, rand() % 100+100);
		}
	return weights;
}

std::shared_ptr<GlsWeightSetCuda> copyWeights(std::shared_ptr<GlsWeightSet> originalWeights, int count) {
	std::shared_ptr<GlsWeightSetCuda> weightsCuda = std::shared_ptr<GlsWeightSetCuda>(new GlsWeightSetCuda(count));
	for (int i = 0; i < count; i++)
		for (int j = 0; j < count; j++) {
			if (i == j) continue;
			weightsCuda->addWeight(i, j, originalWeights->getWeight(i,j));
		}
	weightsCuda->updateCudaWeights();
	return weightsCuda;
}

void generateRandomSolutions(std::shared_ptr<RasterPackingProblem> rasterProblem, int length, int repetitions) {
	std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapEvaluator = std::shared_ptr<RasterTotalOverlapMapEvaluatorGLS>(new RasterTotalOverlapMapEvaluatorGLS(rasterProblem, std::shared_ptr<GlsNoWeightSet>(new GlsNoWeightSet), false));
	std::shared_ptr<RasterStripPackingCompactor> compactor = std::shared_ptr<RasterStripPackingCompactor>(new RasterStripPackingCompactor(length, rasterProblem, overlapEvaluator, 0.04, 0.01));
	for (int i = 0; i < repetitions; i++) {
		std::shared_ptr<RasterPackingSolution> cursolution = std::shared_ptr<RasterPackingSolution>(new RasterPackingSolution(rasterProblem->count()));
		compactor->generateRandomSolution(*cursolution);
		solutions.push_back(cursolution);
		//solution.exportToPgf("sol.pgf", rasterProblem, (qreal)compactor->getCurrentLength() / rasterProblem->getScale(), (qreal)compactor->getCurrentHeight() / rasterProblem->getScale());
	}
}

long long measureOverlapEvaluatorTime(std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapEvaluator, int count) {
	auto start = std::chrono::system_clock::now();
	std::shared_ptr<TotalOverlapMap> curMap;
	for (int i = 0; i < solutions.size(); i++) {
		for (int k = 0; k < count; k++) {
			curMap = overlapEvaluator->getTotalOverlapMap(k, solutions[i]->getOrientation(k), *solutions[i]);
		}
	}
	auto end = std::chrono::system_clock::now();
	//curMap->getImage().save("map" + QString::number(mapcount++) + ".png");
	return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

int main(int argc, char *argv[])
{
	// --> Parse command line arguments
	args::ArgumentParser parser("Raster Packing cuda tester.", "No comments.");
	args::HelpFlag help(parser, "help", "Display this help menu", { 'h', "help" });
	args::CompletionFlag completion(parser, { "complete" });
	args::Positional<std::string> argPuzzle(parser, "source", "Input problem file path");
	args::Positional<int> argLength(parser, "length", "Container length");
	try { parser.ParseCLI(argc, argv); }
	catch (args::Completion e) { std::cout << e.what(); return 0; }
	catch (args::Help) { std::cout << parser; return 0; }
	catch (args::ParseError e) { std::cerr << e.what() << std::endl << parser; return 1; }

	//qsrand(4939495);
	qsrand(QTime::currentTime().msec());

	// Load problem on the CPU
	QString originalPath = QDir::currentPath();
	QString  fileName = QString::fromStdString(args::get(argPuzzle));
	QDir::setCurrent(QFileInfo(fileName).absolutePath());
	RASTERPACKING::PackingProblem problem;
	if (!problem.load(fileName)) { std::cerr << "Could not read puzzle file " << fileName.toStdString() << "." << std::endl; return 1; }
	std::shared_ptr<RasterPackingProblem> rasterProblem = std::shared_ptr<RasterPackingProblem>(new RasterPackingProblem(problem));
	QDir::setCurrent(originalPath);

	// Generate random weights
	std::shared_ptr<GlsWeightSet> weights = generateRandomWeigths(rasterProblem->count());
	// Generate random solutions
	generateRandomSolutions(rasterProblem, args::get(argLength), REPETITIONS);

	// 1 - Default creation	
	std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapEvaluator = std::shared_ptr<RasterTotalOverlapMapEvaluatorGLS>(new RasterTotalOverlapMapEvaluatorGLS(rasterProblem, weights, false));
	std::shared_ptr<RasterStripPackingCompactor> compactor = std::shared_ptr<RasterStripPackingCompactor>(new RasterStripPackingCompactor(args::get(argLength), rasterProblem, overlapEvaluator, 0.04, 0.01));
	long long serialduration = measureOverlapEvaluatorTime(overlapEvaluator, rasterProblem->count());
	std::cout << "Total elapsed time for " << REPETITIONS << " repetitions of default method was " << serialduration << "us." << std::endl;

	// 2 - Full version
	std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapFullEvaluator = std::shared_ptr<RasterTotalOverlapMapEvaluatorFull>(new RasterTotalOverlapMapEvaluatorFull(rasterProblem, weights));
	std::shared_ptr<RasterStripPackingCompactor> compactorFull = std::shared_ptr<RasterStripPackingCompactor>(new RasterStripPackingCompactor(args::get(argLength), rasterProblem, overlapFullEvaluator, 0.04, 0.01));
	long long fullduration = measureOverlapEvaluatorTime(overlapFullEvaluator, rasterProblem->count());
	std::cout << "Total elapsed time for " << REPETITIONS << " repetitions of full method was " << fullduration << "us. Speedup was " << (float)serialduration/(float)fullduration << "." << std::endl;

	// 3 - Matrix version
	std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapMatrixEvaluator = std::shared_ptr<RasterTotalOverlapMapEvaluatorMatrixGLS>(new RasterTotalOverlapMapEvaluatorMatrixGLS(rasterProblem, weights, false));
	std::shared_ptr<RasterStripPackingCompactor> compactorMatrix = std::shared_ptr<RasterStripPackingCompactor>(new RasterStripPackingCompactor(args::get(argLength), rasterProblem, overlapMatrixEvaluator, 0.04, 0.01));
	long long matrixduration = measureOverlapEvaluatorTime(overlapMatrixEvaluator, rasterProblem->count());
	std::cout << "Total elapsed time for " << REPETITIONS << " repetitions of matrix method was " << matrixduration<< "us. Speedup was " << (float)serialduration/(float)matrixduration << "." << std::endl;

	// Load problem on the GPU
	QDir::setCurrent(QFileInfo(fileName).absolutePath());
	RASTERPACKING::PackingProblem problemCuda;
	problemCuda.load(fileName);
	std::shared_ptr<RasterPackingProblem> rasterCudaProblem = std::shared_ptr<RasterPackingProblem>(new RasterPackingCudaProblem(problemCuda));
	QDir::setCurrent(originalPath);
	std::shared_ptr<GlsWeightSetCuda> weightsCuda = copyWeights(weights, rasterProblem->count());

	// 4 - Cuda version
	std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapCudaEvaluator = std::shared_ptr<RasterTotalOverlapMapEvaluatorCudaGLS>(new RasterTotalOverlapMapEvaluatorCudaGLS(rasterCudaProblem, weights));
	std::shared_ptr<RasterStripPackingCompactor> compactorCuda = std::shared_ptr<RasterStripPackingCompactor>(new RasterStripPackingCompactor(args::get(argLength), rasterCudaProblem, overlapCudaEvaluator, 0.04, 0.01));
	long long cudaduration = measureOverlapEvaluatorTime(overlapCudaEvaluator, rasterProblem->count());
	std::cout << "Total elapsed time for " << REPETITIONS << " repetitions of cuda method was " << cudaduration << "us. Speedup was " << (float)serialduration / (float)cudaduration << "." << std::endl;

	// 5 - Cuda matrix version
	std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapCudaMatrixEvaluator = std::shared_ptr<RasterTotalOverlapMapEvaluatorCudaMatrixGLS>(new RasterTotalOverlapMapEvaluatorCudaMatrixGLS(rasterCudaProblem, weightsCuda, false));
	std::shared_ptr<RasterStripPackingCompactor> compactorCudaMatrix = std::shared_ptr<RasterStripPackingCompactor>(new RasterStripPackingCompactor(args::get(argLength), rasterCudaProblem, overlapCudaMatrixEvaluator, 0.04, 0.01));
	long long cudamatrixduration = measureOverlapEvaluatorTime(overlapCudaMatrixEvaluator, rasterProblem->count());
	std::cout << "Total elapsed time for " << REPETITIONS << " repetitions of cuda matrix method was " << cudamatrixduration << "us. Speedup was " << (float) serialduration / (float)cudamatrixduration << "." << std::endl;

	return 0;
}
