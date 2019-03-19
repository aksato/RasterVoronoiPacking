#include "args/args.hxx"
#include "packingproblem.h"
#include "raster/rasterpackingproblem.h"
#include "cuda/rasterpackingcudaproblem.h"
#include "cuda/rasteroverlapevaluatorcudagls.h"
#include "cuda/rasteroverlapevaluatorcudamatrixgls.h"
#include "cuda/glsweightsetcuda.h"
#include "raster/rasteroverlapevaluator.h"
#include "raster/rasteroverlapevaluatorfull.h"
#include "raster/rasteroverlapevaluatormatrixgls.h"
#include "raster/rasterstrippackingparameters.h"
#include "raster/rasterstrippackingsolver.h"
#include "raster/rastersquarepackingcompactor.h"
#include <fstream>
#include <iostream>
#include <QString>
#include <QDir>
#include <QFileInfo>
#include <QTime>

using namespace RASTERVORONOIPACKING;

QVector<std::shared_ptr<RasterPackingSolution>> solutions;

bool fileExists(const char *fileName)
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

void generateRandomSolutions(std::shared_ptr<RasterPackingProblem> rasterProblem, int length, int repetitions, bool outputsolutions) {
	std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapEvaluator = std::shared_ptr<RasterTotalOverlapMapEvaluatorGLS>(new RasterTotalOverlapMapEvaluatorGLS(rasterProblem, std::shared_ptr<GlsNoWeightSet>(new GlsNoWeightSet), false));
	std::shared_ptr<RasterStripPackingCompactor> compactor = std::shared_ptr<RasterStripPackingCompactor>(new RasterStripPackingCompactor(length, rasterProblem, overlapEvaluator, 0.04, 0.01));
	for (int i = 0; i < repetitions; i++) {
		std::shared_ptr<RasterPackingSolution> cursolution = std::shared_ptr<RasterPackingSolution>(new RasterPackingSolution(rasterProblem->count()));
		compactor->generateRandomSolution(*cursolution);
		solutions.push_back(cursolution);
		if(outputsolutions) cursolution->exportToPgf("sol" + QString::number(i+1) + ".pgf", rasterProblem, (qreal)compactor->getCurrentLength() / rasterProblem->getScale(), (qreal)compactor->getCurrentHeight() / rasterProblem->getScale());
	}
}

void saveMap(std::shared_ptr<TotalOverlapMap> map, QString basename, int id, bool cuda) {
	if (!cuda) map->getImage().save("map" + basename + QString::number(id) + ".png");
	else {
		std::shared_ptr<TotalOverlapMap> dummyPieceMap = std::shared_ptr<TotalOverlapMap>(new TotalOverlapMap(map->getRect(), map->getCuttingStockLength()));
		cudaMemcpy(dummyPieceMap->getData(), map->getData(), map->getHeight() * map->getWidth() * sizeof(quint32), cudaMemcpyDeviceToHost);
		dummyPieceMap->getImage().save("map" + basename + QString::number(id) + ".png");
	}
}

long long measureOverlapEvaluatorTime(std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapEvaluator, QString overlapEvaluatorName, bool outputmaps, bool cuda = false) {
	int mapcount = 1;
	auto start = std::chrono::system_clock::now();
	std::shared_ptr<TotalOverlapMap> curMap;
	for (int i = 0; i < solutions.size(); i++) {
		for (int k = 0; k < overlapEvaluator->problem->count(); k++) {
			curMap = overlapEvaluator->getTotalOverlapMap(k, solutions[i]->getOrientation(k), *solutions[i]);
			if (outputmaps) saveMap(curMap, overlapEvaluatorName, mapcount++, cuda);
		}
	}
	if(cuda) cudaDeviceSynchronize();
	auto end = std::chrono::system_clock::now();
	long long duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "Total elapsed time for " << solutions.size() << " repetitions of " << overlapEvaluatorName.toStdString() << " method was " << duration << "us.";
	return duration;
}

auto computeProblemStatistics(std::shared_ptr<RasterPackingProblem> rasterProblem) {
	double avgNfpSize, avgIfpSize;
	avgIfpSize = 0; avgNfpSize = 0;
	int ifpCount = 0;
	for (int itemTypeId = 0; itemTypeId < rasterProblem->getItemTypeCount(); itemTypeId++) {
		for (uint angle = 0; angle < (*rasterProblem->getItemByType(itemTypeId))->getAngleCount(); angle++) {
			std::shared_ptr<RasterNoFitPolygon> curIfp = rasterProblem->getIfps()->getRasterNoFitPolygon(0, 0, itemTypeId, angle);
			avgIfpSize += curIfp->width() * curIfp->height() * rasterProblem->getMultiplicity(itemTypeId);
			ifpCount += rasterProblem->getMultiplicity(itemTypeId);
		}
	}
	avgIfpSize /= ifpCount;
	int nfpCount = 0;
	for (int item1TypeId = 0; item1TypeId < rasterProblem->getItemTypeCount(); item1TypeId++)
		for (uint angle1 = 0; angle1 < (*rasterProblem->getItemByType(item1TypeId))->getAngleCount(); angle1++)
			for (int item2TypeId = item1TypeId + 1; item2TypeId < rasterProblem->getItemTypeCount(); item2TypeId++)
				for (uint angle2 = 0; angle2 < (*rasterProblem->getItemByType(item2TypeId))->getAngleCount(); angle2++) {
					std::shared_ptr<RasterNoFitPolygon> curNfp = rasterProblem->getNfps()->getRasterNoFitPolygon(item1TypeId, angle1, item2TypeId, angle2);
					avgNfpSize += curNfp->width() * curNfp->height() * rasterProblem->getMultiplicity(item1TypeId) * rasterProblem->getMultiplicity(item2TypeId);
					nfpCount += rasterProblem->getMultiplicity(item1TypeId) * rasterProblem->getMultiplicity(item2TypeId);
				}
	avgNfpSize /= nfpCount;
	return std::make_tuple(avgNfpSize, avgIfpSize);
}

int main(int argc, char *argv[])
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

	qsrand(QTime::currentTime().msec());
	if(argSeed) qsrand(args::get(argSeed));
	int repetitions = args::get(argRepetitions);

	// Load problem on the CPU
	QString originalPath = QDir::currentPath();
	QString  fileName = QString::fromStdString(args::get(argPuzzle));
	fileName = QFileInfo(fileName).absoluteFilePath();
	QDir::setCurrent(QFileInfo(fileName).absolutePath());
	RASTERPACKING::PackingProblem problem;
	if (!problem.load(fileName)) { std::cerr << "Could not read puzzle file " << fileName.toStdString() << "." << std::endl; return 1; }
	std::shared_ptr<RasterPackingProblem> rasterProblem = std::shared_ptr<RasterPackingProblem>(new RasterPackingProblem(problem));
	QDir::setCurrent(originalPath);

	// Generate random weights
	std::shared_ptr<GlsWeightSet> weights = generateRandomWeigths(rasterProblem->count());
	// Generate random solutions
	int length = args::get(argLength) * rasterProblem->getScale();
	generateRandomSolutions(rasterProblem, length, repetitions, argOutputImages);
	// Compute statistics
	int avgNfpSize, avgIfpSize;
	std::tie(avgNfpSize, avgIfpSize) = computeProblemStatistics(rasterProblem);

	// 1 - Default creation	
	std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapEvaluator = std::shared_ptr<RasterTotalOverlapMapEvaluatorGLS>(new RasterTotalOverlapMapEvaluatorGLS(rasterProblem, weights, false));
	std::shared_ptr<RasterStripPackingCompactor> compactor = std::shared_ptr<RasterStripPackingCompactor>(new RasterStripPackingCompactor(length, rasterProblem, overlapEvaluator, 0.04, 0.01));
	long long serialduration = measureOverlapEvaluatorTime(overlapEvaluator, "serial", argOutputImages); std::cout << std::endl;

	long long fullduration;
	if (testFullMethod) {
		// 2 - Full version
		std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapFullEvaluator = std::shared_ptr<RasterTotalOverlapMapEvaluatorFull>(new RasterTotalOverlapMapEvaluatorFull(rasterProblem, weights));
		std::shared_ptr<RasterStripPackingCompactor> compactorFull = std::shared_ptr<RasterStripPackingCompactor>(new RasterStripPackingCompactor(length, rasterProblem, overlapFullEvaluator, 0.04, 0.01));
		fullduration = measureOverlapEvaluatorTime(overlapFullEvaluator, "full", argOutputImages);
		std::cout << " Speedup was " << (float)serialduration / (float)fullduration << "." << std::endl;
	}

	long long matrixduration;
	if(!argIgnoreMatrix) {
		// 3 - Matrix version
		std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapMatrixEvaluator = std::shared_ptr<RasterTotalOverlapMapEvaluatorMatrixGLS>(new RasterTotalOverlapMapEvaluatorMatrixGLS(rasterProblem, weights, false));
		std::shared_ptr<RasterStripPackingCompactor> compactorMatrix = std::shared_ptr<RasterStripPackingCompactor>(new RasterStripPackingCompactor(length, rasterProblem, overlapMatrixEvaluator, 0.04, 0.01));
		matrixduration = measureOverlapEvaluatorTime(overlapMatrixEvaluator, "matrix", argOutputImages);
		std::cout << " Speedup was " << (float)serialduration/(float)matrixduration << "." << std::endl;
	}

	// Load problem on the GPU
	QDir::setCurrent(QFileInfo(fileName).absolutePath());
	RASTERPACKING::PackingProblem problemCuda;
	problemCuda.load(fileName);
	std::shared_ptr<RasterPackingProblem> rasterCudaProblem = std::shared_ptr<RasterPackingProblem>(new RasterPackingCudaProblem(problemCuda));
	QDir::setCurrent(originalPath);
	std::shared_ptr<GlsWeightSetCuda> weightsCuda = copyWeights(weights, rasterProblem->count());

	// 4 - Cuda version
	std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapCudaEvaluator = std::shared_ptr<RasterTotalOverlapMapEvaluatorCudaGLS>(new RasterTotalOverlapMapEvaluatorCudaGLS(rasterCudaProblem, weights));
	std::shared_ptr<RasterStripPackingCompactor> compactorCuda = std::shared_ptr<RasterStripPackingCompactor>(new RasterStripPackingCompactor(length, rasterCudaProblem, overlapCudaEvaluator, 0.04, 0.01));
	long long cudaduration = measureOverlapEvaluatorTime(overlapCudaEvaluator, "cuda", argOutputImages, true);
	std::cout << " Speedup was " << (float)serialduration / (float)cudaduration << "." << std::endl;

	long long cudamatrixduration;
	if (!argIgnoreMatrix) {
		// 5 - Cuda matrix version
		std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapCudaMatrixEvaluator = std::shared_ptr<RasterTotalOverlapMapEvaluatorCudaMatrixGLS>(new RasterTotalOverlapMapEvaluatorCudaMatrixGLS(rasterCudaProblem, weightsCuda, false));
		std::shared_ptr<RasterStripPackingCompactor> compactorCudaMatrix = std::shared_ptr<RasterStripPackingCompactor>(new RasterStripPackingCompactor(length, rasterCudaProblem, overlapCudaMatrixEvaluator, 0.04, 0.01));
		cudamatrixduration = measureOverlapEvaluatorTime(overlapCudaMatrixEvaluator, "matcuda", argOutputImages, true);
		std::cout << " Speedup was " << (float)serialduration / (float)cudamatrixduration << "." << std::endl;
	}

	// Print report of results
	if (argReport) {
		std::ofstream outfile;
		if (!fileExists(args::get(argReport).c_str())) {
			outfile.open(args::get(argReport));
			outfile << "Case, Scale, Items, Avg Ifp Size, Avg Nfp Size, Serial" << (!argIgnoreMatrix ? ", Matrix" : "") << (testFullMethod ? ", Full" : "") << ", Cuda" << (!argIgnoreMatrix ? ", Cuda Matrix" : "") << std::endl;
			outfile.close();
		}
		outfile.open(args::get(argReport), std::ios_base::app);
		outfile << QFileInfo(fileName).baseName().toStdString() << "," << rasterProblem->getScale() << "," << rasterProblem->count() << "," << avgIfpSize << "," << avgNfpSize;
		outfile << "," << serialduration;
		if (!argIgnoreMatrix) outfile << "," << matrixduration;
		if (testFullMethod) outfile << "," << fullduration;
		outfile << "," << cudaduration ;
		if (!argIgnoreMatrix) outfile << "," << cudamatrixduration;
		outfile << std::endl;
		outfile.close();
	}

	return 0;
}
