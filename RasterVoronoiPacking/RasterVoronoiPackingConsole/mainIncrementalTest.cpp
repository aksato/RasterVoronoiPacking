#include "args/args.hxx"
#include "packingproblem.h"
#include "raster/rasterpackingproblem.h"
#include "raster/rasteroverlapevaluator.h"
#include "raster/rasteroverlapevaluatorincremental.h"
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

std::shared_ptr<RasterPackingProblem> rasterProblem;
int repetitions;

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

void saveMap(std::shared_ptr<TotalOverlapMap> map, QString basename, int id) {
	map->getImage().save("map" + basename + QString::number(id) + ".png");
}

long long measureOverlapEvaluatorTime(std::shared_ptr<RasterStripPackingSolver> solver, QString overlapEvaluatorName, 
	std::shared_ptr<RasterStripPackingCompactor> compactor, bool outputsols, bool cuda = false) {
	
	std::shared_ptr<GlsWeightSet> weights = generateRandomWeigths(rasterProblem->count());
	RASTERVORONOIPACKING::RasterPackingSolution solution(rasterProblem->count());
	QVector<quint32> currentOverlaps(rasterProblem->count() * rasterProblem->count());
	quint32 maxItemOverlap;
	compactor->generateRandomSolution(solution);

	auto start = std::chrono::system_clock::now();
	for (int k = 0; k < repetitions; k++) {
		solver->performLocalSearch(solution);
		if(outputsols)
			solution.exportToPgf("sol" + overlapEvaluatorName + QString::number(k + 1) + ".pgf", rasterProblem, (qreal)compactor->getCurrentLength() / rasterProblem->getScale(), (qreal)compactor->getCurrentHeight() / rasterProblem->getScale());
		solver->getGlobalOverlap(solution, currentOverlaps, maxItemOverlap);
		solver->updateWeights(solution, currentOverlaps, maxItemOverlap);
	}
	auto end = std::chrono::system_clock::now();
	long long duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "Total elapsed time for " << repetitions << " repetitions of " << overlapEvaluatorName.toStdString() << " method was " << duration << "us.";
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

	// 1 - Default creation	
	qsrand(seed);
	std::shared_ptr<GlsWeightSet> weights = generateRandomWeigths(rasterProblem->count());
	std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapEvaluator = std::shared_ptr<RasterTotalOverlapMapEvaluatorGLS>(new RasterTotalOverlapMapEvaluatorGLS(rasterProblem, weights, false));
	std::shared_ptr<RasterStripPackingCompactor> compactor = std::shared_ptr<RasterStripPackingCompactor>(new RasterStripPackingCompactor(length, rasterProblem, overlapEvaluator, 0.04, 0.01));
	std::shared_ptr<RasterStripPackingSolver> solver(new RasterStripPackingSolver(rasterProblem, overlapEvaluator));
	long long serialduration = measureOverlapEvaluatorTime(solver, "serial", compactor, argOutputImages); std::cout << std::endl;

	// 2 - Incremental creation
	qsrand(seed);
	weights = generateRandomWeigths(rasterProblem->count());
	std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapIncEvaluator = std::shared_ptr<RasterTotalOverlapMapEvaluatorIncremental>(new RasterTotalOverlapMapEvaluatorIncremental(rasterProblem, weights, false));
	std::shared_ptr<RasterStripPackingCompactor> compactorInc = std::shared_ptr<RasterStripPackingCompactor>(new RasterStripPackingCompactor(length, rasterProblem, overlapIncEvaluator, 0.04, 0.01));
	std::shared_ptr<RasterStripPackingSolver> solverInc(new RasterStripPackingSolver(rasterProblem, overlapIncEvaluator));
	long long serialincduration = measureOverlapEvaluatorTime(solverInc, "serialinc", compactorInc, argOutputImages); std::cout << std::endl;
	std::cout << " Speedup was " << (float)serialduration / (float)serialincduration << "." << std::endl;

	return 0;
}
