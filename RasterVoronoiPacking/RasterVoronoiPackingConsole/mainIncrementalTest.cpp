#include "args/args.hxx"
#include "packingproblem.h"
#include "raster/rasterpackingproblem.h"
#include "raster/rasteroverlapevaluator.h"
#include "raster/rasteroverlapevaluatorincremental.h"
#include "raster/rasteroverlapevaluatorfull.h"
#include "raster/rasteroverlapevaluatormatrixgls.h"
#include "raster/rasterstrippackingparameters.h"
#include "raster/rasterstrippackingsolver.h"
#include "raster/rastersquarepackingcompactor.h"
#include "cuda/rasterpackingcudaproblem.h"
#include "cuda/rasteroverlapevaluatorcudagls.h"
#include "cuda/rasteroverlapevaluatorcudainc.h"
#include "cuda/rasteroverlapevaluatorcudafull.h"
#include "cuda/rasteroverlapevaluatorcudamatrixgls.h"
#include "cuda/glsweightsetcuda.h"
#include <fstream>
#include <iostream>
#include <QString>
#include <QDir>
#include <QFileInfo>
#include <QTime>

using namespace RASTERVORONOIPACKING;

std::shared_ptr<RasterPackingProblem> rasterProblem;
int repetitions;

bool fileExists(const char* fileName);
std::shared_ptr<GlsWeightSet> generateRandomWeigths(int count);
long long measureOverlapEvaluatorTime(std::shared_ptr<RasterStripPackingSolver> solver, QString overlapEvaluatorName, std::shared_ptr<RasterStripPackingCompactor> compactor, bool outputsols, bool cuda = false);
std::shared_ptr<GlsWeightSetCuda> copyWeights(std::shared_ptr<GlsWeightSet> originalWeights, int count);

int main(int argc, char* argv[])
{
	// --> Parse command line arguments
	args::ArgumentParser parser("Raster Packing cuda tester.", "No comments.");
	args::HelpFlag help(parser, "help", "Display this help menu", { 'h', "help" });
	args::CompletionFlag completion(parser, { "complete" });
	args::ValueFlag<int> argSeed(parser, "value", "Manual seed input for the random number generator (for debugging purposes)", { "seed" });
	args::ValueFlag<std::string> argReport(parser, "filename", "Name of generated report of overlap evaluators performance (elapsed time in us) in the following order: serial, full, matrix, cuda, cuda matrix", { "report" });
	args::Flag argOutputImages(parser, "flag", "Output overlap map images and generated random layouts", { "output" });
	args::Flag argIgnoreMatrix(parser, "flag", "Skip evaluation of matrix overlap evaluator methods", { "skipmatrix" });
	args::Flag testFullMethod(parser, "flag", "Evaluate full method (very inefficient)", { "testfull" });
	args::Positional<std::string> argPuzzle(parser, "source", "Input problem file path");
	args::Positional<int> argLength(parser, "length", "Container length");
	args::Positional<int> argRepetitions(parser, "repetitions", "Number of overlap map creator executions");
	try { parser.ParseCLI(argc, argv); }
	catch (args::Completion e) { std::cout << e.what(); return 0; }
	catch (args::Help) { std::cout << parser; return 0; }
	catch (args::ParseError e) { std::cerr << e.what() << std::endl << parser; return 1; }

	//qsrand(QTime::currentTime().msec());
	int seed = argSeed ? args::get(argSeed) : QTime::currentTime().msec();
	//if (argSeed) qsrand(args::get(argSeed));
	repetitions = args::get(argRepetitions);

	// Load problem on the CPU
	QString originalPath = QDir::currentPath();
	QString  fileName = QString::fromStdString(args::get(argPuzzle));
	fileName = QFileInfo(fileName).absoluteFilePath();
	QDir::setCurrent(QFileInfo(fileName).absolutePath());
	RASTERPACKING::PackingProblem problem;
	if (!problem.load(fileName)) { std::cerr << "Could not read puzzle file " << fileName.toStdString() << "." << std::endl; return 1; }
	rasterProblem = std::shared_ptr<RasterPackingProblem>(new RasterPackingProblem(problem));
	int length = args::get(argLength) * rasterProblem->getScale();
	QDir::setCurrent(originalPath);

	long long serialduration;
	{
		// 1.1 - Default creation
		qsrand(seed);
		std::shared_ptr<GlsWeightSet> weights = generateRandomWeigths(rasterProblem->count());
		std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapEvaluator = std::shared_ptr<RasterTotalOverlapMapEvaluatorGLS>(new RasterTotalOverlapMapEvaluatorGLS(rasterProblem, weights, false));
		std::shared_ptr<RasterStripPackingCompactor> compactor = std::shared_ptr<RasterStripPackingCompactor>(new RasterStripPackingCompactor(length, rasterProblem, overlapEvaluator, 0.04, 0.01));
		std::shared_ptr<RasterStripPackingSolver> solver(new RasterStripPackingSolver(rasterProblem, overlapEvaluator));
		serialduration = measureOverlapEvaluatorTime(solver, "serial", compactor, argOutputImages); std::cout << std::endl;
	}

	long long serialincduration;
	{
		// 1.2 - Incremental creation
		qsrand(seed);
		std::shared_ptr<GlsWeightSet> weights = generateRandomWeigths(rasterProblem->count());
		std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapIncEvaluator = std::shared_ptr<RasterTotalOverlapMapEvaluatorIncremental>(new RasterTotalOverlapMapEvaluatorIncremental(rasterProblem, weights, false));
		std::shared_ptr<RasterStripPackingCompactor> compactorInc = std::shared_ptr<RasterStripPackingCompactor>(new RasterStripPackingCompactor(length, rasterProblem, overlapIncEvaluator, 0.04, 0.01));
		std::shared_ptr<RasterStripPackingSolver> solverInc(new RasterStripPackingSolver(rasterProblem, overlapIncEvaluator));
		serialincduration = measureOverlapEvaluatorTime(solverInc, "serialinc", compactorInc, argOutputImages);
		std::cout << " Speedup was " << (float)serialduration / (float)serialincduration << "." << std::endl;
	}

	long long cacheduration;
	{
		// 1.3 - Cache creation
		qsrand(seed);
		std::shared_ptr<GlsWeightSet> weights = generateRandomWeigths(rasterProblem->count());
		std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapEvaluatorCache = std::shared_ptr<RasterTotalOverlapMapEvaluatorGLS>(new RasterTotalOverlapMapEvaluatorGLS(rasterProblem, weights, true));
		std::shared_ptr<RasterStripPackingCompactor> compactorCache = std::shared_ptr<RasterStripPackingCompactor>(new RasterStripPackingCompactor(length, rasterProblem, overlapEvaluatorCache, 0.04, 0.01));
		std::shared_ptr<RasterStripPackingSolver> solverCache(new RasterStripPackingSolver(rasterProblem, overlapEvaluatorCache));
		cacheduration = measureOverlapEvaluatorTime(solverCache, "cache", compactorCache, argOutputImages);
		std::cout << " Speedup was " << (float)serialduration / (float)cacheduration << "." << std::endl;
	}

	long long fullduration;
	{
		// 1.4 - Full creation
		qsrand(seed);
		std::shared_ptr<GlsWeightSet> weights = generateRandomWeigths(rasterProblem->count());
		std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapEvaluatorFull = std::shared_ptr<RasterTotalOverlapMapEvaluatorFull>(new RasterTotalOverlapMapEvaluatorFull(rasterProblem, weights, true));
		std::shared_ptr<RasterStripPackingCompactor> compactorFull = std::shared_ptr<RasterStripPackingCompactor>(new RasterStripPackingCompactor(length, rasterProblem, overlapEvaluatorFull, 0.04, 0.01));
		std::shared_ptr<RasterStripPackingSolver> solverFull(new RasterStripPackingSolver(rasterProblem, overlapEvaluatorFull));
		fullduration = measureOverlapEvaluatorTime(solverFull, "full", compactorFull, argOutputImages);
		std::cout << " Speedup was " << (float)serialduration / (float)fullduration << "." << std::endl;
	}

	long long matrixduration;
	// 1.5 - Matrix version
	{
		qsrand(seed);
		std::shared_ptr<GlsWeightSet> weights = generateRandomWeigths(rasterProblem->count());
		std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapEvaluatorMatrix = std::shared_ptr<RasterTotalOverlapMapEvaluatorMatrixGLS>(new RasterTotalOverlapMapEvaluatorMatrixGLS(rasterProblem, weights, true));
		std::shared_ptr<RasterStripPackingCompactor> compactorMatrix = std::shared_ptr<RasterStripPackingCompactor>(new RasterStripPackingCompactor(length, rasterProblem, overlapEvaluatorMatrix, 0.04, 0.01));
		std::shared_ptr<RasterStripPackingSolver> solverMatrix(new RasterStripPackingSolver(rasterProblem, overlapEvaluatorMatrix));
		matrixduration = measureOverlapEvaluatorTime(solverMatrix, "matrix", compactorMatrix, argOutputImages);
		std::cout << " Speedup was " << (float)serialduration / (float)matrixduration << "." << std::endl;
	}

	// Load problem on the GPU
	QDir::setCurrent(QFileInfo(fileName).absolutePath());
	RASTERPACKING::PackingProblem problemCuda;
	problemCuda.load(fileName);
	std::shared_ptr<RasterPackingProblem> rasterCudaProblem = std::shared_ptr<RasterPackingProblem>(new RasterPackingCudaProblem(problemCuda));
	QDir::setCurrent(originalPath);

	long long cudaduration;
	{
		// 1.1 - Cuda version
		qsrand(seed);
		std::shared_ptr<GlsWeightSet> weights = generateRandomWeigths(rasterCudaProblem->count());
		std::shared_ptr<GlsWeightSetCuda> weightsCuda = copyWeights(weights, rasterCudaProblem->count());
		std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapCudaEvaluator = std::shared_ptr<RasterTotalOverlapMapEvaluatorCudaGLS>(new RasterTotalOverlapMapEvaluatorCudaGLS(rasterCudaProblem, weightsCuda));
		std::shared_ptr<RasterStripPackingCompactor> compactorCuda = std::shared_ptr<RasterStripPackingCompactor>(new RasterStripPackingCompactor(length, rasterCudaProblem, overlapCudaEvaluator, 0.04, 0.01));
		std::shared_ptr<RasterStripPackingSolver> solverCuda(new RasterStripPackingSolver(rasterProblem, overlapCudaEvaluator));
		cudaduration = measureOverlapEvaluatorTime(solverCuda, "cuda", compactorCuda, argOutputImages);
		std::cout << " Speedup was " << (float)serialduration / (float)cudaduration << "." << std::endl;
	}

	long long cudaincduration;
	{
		// 1.2 - Incremental Cuda version
		qsrand(seed);
		std::shared_ptr<GlsWeightSet> weights = generateRandomWeigths(rasterCudaProblem->count());
		std::shared_ptr<GlsWeightSetCuda> weightsCudaInc = copyWeights(weights, rasterCudaProblem->count());
		std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapCudaIncEvaluator = std::shared_ptr<RasterTotalOverlapMapEvaluatorCudaIncremental>(new RasterTotalOverlapMapEvaluatorCudaIncremental(rasterCudaProblem, weightsCudaInc));
		std::shared_ptr<RasterStripPackingCompactor> compactorCudaInc = std::shared_ptr<RasterStripPackingCompactor>(new RasterStripPackingCompactor(length, rasterCudaProblem, overlapCudaIncEvaluator, 0.04, 0.01));
		std::shared_ptr<RasterStripPackingSolver> solverCudaInc(new RasterStripPackingSolver(rasterProblem, overlapCudaIncEvaluator));
		cudaincduration = measureOverlapEvaluatorTime(solverCudaInc, "cudainc", compactorCudaInc, argOutputImages);
		std::cout << " Speedup was " << (float)serialduration / (float)cudaincduration << "." << std::endl;
	}

	long long cudafullduration;
	{
		// 1.4 - Full Cuda version
		qsrand(seed);
		std::shared_ptr<GlsWeightSet> weights = generateRandomWeigths(rasterCudaProblem->count());
		std::shared_ptr<GlsWeightSetCuda> weightsCudaFull = copyWeights(weights, rasterCudaProblem->count());
		std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapCudaFullEvaluator = std::shared_ptr<RasterTotalOverlapMapEvaluatorCudaFull>(new RasterTotalOverlapMapEvaluatorCudaFull(rasterCudaProblem, weightsCudaFull));
		std::shared_ptr<RasterStripPackingCompactor> compactorCudaFull = std::shared_ptr<RasterStripPackingCompactor>(new RasterStripPackingCompactor(length, rasterCudaProblem, overlapCudaFullEvaluator, 0.04, 0.01));
		std::shared_ptr<RasterStripPackingSolver> solverCudaFull(new RasterStripPackingSolver(rasterProblem, overlapCudaFullEvaluator));
		cudafullduration = measureOverlapEvaluatorTime(solverCudaFull, "cudafull", compactorCudaFull, argOutputImages);
		std::cout << " Speedup was " << (float)serialduration / (float)cudafullduration << "." << std::endl;
	}

	long long cudamatrixduration;
	{
		// 1.5 - Matrix Cuda version
		qsrand(seed);
		std::shared_ptr<GlsWeightSet> weights = generateRandomWeigths(rasterCudaProblem->count());
		std::shared_ptr<GlsWeightSetCuda> weightsCudaFull = copyWeights(weights, rasterCudaProblem->count());
		std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapCudaMatrixEvaluator = std::shared_ptr<RasterTotalOverlapMapEvaluatorCudaMatrixGLS>(new RasterTotalOverlapMapEvaluatorCudaMatrixGLS(rasterCudaProblem, weightsCudaFull, false));
		std::shared_ptr<RasterStripPackingCompactor> compactorCudaMatrix = std::shared_ptr<RasterStripPackingCompactor>(new RasterStripPackingCompactor(length, rasterCudaProblem, overlapCudaMatrixEvaluator, 0.04, 0.01));
		std::shared_ptr<RasterStripPackingSolver> solverCudaMatrix(new RasterStripPackingSolver(rasterProblem, overlapCudaMatrixEvaluator));
		cudamatrixduration = measureOverlapEvaluatorTime(solverCudaMatrix, "cudamatrix", compactorCudaMatrix, argOutputImages);
		std::cout << " Speedup was " << (float)serialduration / (float)cudamatrixduration << "." << std::endl;
	}

	// Print report of results
	if (argReport) {
		std::ofstream outfile;
		if (!fileExists(args::get(argReport).c_str())) {
			outfile.open(args::get(argReport));
			outfile << "Case, Scale, Serial, Increment, Full, Matrix, Cuda, CudaInc, CudaFull, CudaMat" << std::endl;
			outfile.close();
		}
		outfile.open(args::get(argReport), std::ios_base::app);
		outfile << QFileInfo(fileName).baseName().toStdString() << "," << rasterProblem->getScale() <<
			"," << serialduration << "," << serialincduration << "," << fullduration << "," << matrixduration << "," <<
			"," << cudaduration << "," << cudaincduration << "," << cudafullduration << "," << cudamatrixduration << std::endl;
		outfile.close();
	}

	return 0;
}

bool fileExists(const char* fileName)
{
	std::ifstream infile(fileName);
	bool ans = infile.good();
	infile.close();
	return ans;
}

std::shared_ptr<GlsWeightSet> generateRandomWeigths(int count) {
	std::shared_ptr<GlsWeightSet> weights = std::shared_ptr<GlsWeightSet>(new GlsWeightSet(count));
	for (int i = 0; i < count; i++)
		for (int j = 0; j < count; j++) {
			if (i == j) continue;
			weights->addWeight(i, j, rand() % 100 + 100);
		}
	return weights;
}

long long measureOverlapEvaluatorTime(std::shared_ptr<RasterStripPackingSolver> solver, QString overlapEvaluatorName,
	std::shared_ptr<RasterStripPackingCompactor> compactor, bool outputsols, bool cuda) {

	std::shared_ptr<GlsWeightSet> weights = generateRandomWeigths(rasterProblem->count());
	RASTERVORONOIPACKING::RasterPackingSolution solution(rasterProblem->count());
	QVector<quint32> currentOverlaps(rasterProblem->count() * rasterProblem->count());
	quint32 maxItemOverlap;
	compactor->generateRandomSolution(solution);

	auto start = std::chrono::system_clock::now();
	for (int k = 0; k < repetitions; k++) {
		solver->performLocalSearch(solution);
		if (outputsols)
			solution.exportToPgf("sol" + overlapEvaluatorName + QString::number(k + 1) + ".pgf", rasterProblem, (qreal)compactor->getCurrentLength() / rasterProblem->getScale(), (qreal)compactor->getCurrentHeight() / rasterProblem->getScale());
		solver->getGlobalOverlap(solution, currentOverlaps, maxItemOverlap);
		solver->updateWeights(solution, currentOverlaps, maxItemOverlap);
	}
	auto end = std::chrono::system_clock::now();
	long long duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "Total elapsed time for " << repetitions << " repetitions of " << overlapEvaluatorName.toStdString() << " method was " << duration << "us.";
	return duration;
}

std::shared_ptr<GlsWeightSetCuda> copyWeights(std::shared_ptr<GlsWeightSet> originalWeights, int count) {
	std::shared_ptr<GlsWeightSetCuda> weightsCuda = std::shared_ptr<GlsWeightSetCuda>(new GlsWeightSetCuda(count));
	for (int i = 0; i < count; i++)
		for (int j = 0; j < count; j++) {
			if (i == j) continue;
			weightsCuda->addWeight(i, j, originalWeights->getWeight(i, j));
		}
	weightsCuda->updateCudaWeights();
	return weightsCuda;
}