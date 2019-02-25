#include "args/args.hxx"
#include "packingproblem.h"
#include "raster/rasterpackingproblem.h"
#include "raster/rasteroverlapevaluator.h"
#include "raster/rasteroverlapevaluatormatrixgls.h"
#include "raster/rasterstrippackingparameters.h"
#include "raster/rasterstrippackingsolver.h"
#include "raster/rastersquarepackingcompactor.h"
#include <iostream>
#include <QString>
#include <QDir>
#include <QFileInfo>

#define REPETITIONS 1000

using namespace RASTERVORONOIPACKING;

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

	QString originalPath = QDir::currentPath();
	QString  fileName = QString::fromStdString(args::get(argPuzzle));
	QDir::setCurrent(QFileInfo(fileName).absolutePath());
	RASTERPACKING::PackingProblem problem;
	if (!problem.load(fileName)) { std::cerr << "Could not read puzzle file " << fileName.toStdString() << "." << std::endl; return 1; }
	std::shared_ptr<RasterPackingProblem> rasterProblem = std::shared_ptr<RasterPackingProblem>(new RasterPackingProblem(problem));
	RasterPackingSolution	solution = RasterPackingSolution(rasterProblem->count());
	QDir::setCurrent(originalPath);

	qsrand(4939495);
	std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapEvaluator = std::shared_ptr<RasterTotalOverlapMapEvaluatorGLS>(new RasterTotalOverlapMapEvaluatorGLS(rasterProblem, std::shared_ptr<GlsWeightSet>(new GlsWeightSet(rasterProblem->count())))); overlapEvaluator->disableMapCache();
	std::shared_ptr<RasterStripPackingCompactor> compactor = std::shared_ptr<RasterStripPackingCompactor>(new RasterStripPackingCompactor(args::get(argLength), rasterProblem, overlapEvaluator, 0.04, 0.01));
	compactor->generateRandomSolution(solution);
	//solution.exportToPgf("sol.pgf", rasterProblem, (qreal)compactor->getCurrentLength() / rasterProblem->getScale(), (qreal)compactor->getCurrentHeight() / rasterProblem->getScale());

	// Default creation
	auto start = std::chrono::system_clock::now();
	std::shared_ptr<TotalOverlapMap> curMap;
	for (int i = 0; i < REPETITIONS; i++) {
		for (int k = 0; k < rasterProblem->count(); k++) {
			curMap = overlapEvaluator->getTotalOverlapMap(k, solution.getOrientation(k), solution);
		}
	}
	auto end = std::chrono::system_clock::now();
	std::cout << "Total elapsed time for " << REPETITIONS << " repetitions of default method was " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us." << std::endl;
	//curMap->getImage().save("map.png");

	// Matrix version
	std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapMatrixEvaluator = std::shared_ptr<RasterTotalOverlapMapEvaluatorMatrixGLS>(new RasterTotalOverlapMapEvaluatorMatrixGLS(rasterProblem, std::shared_ptr<GlsWeightSet>(new GlsWeightSet(rasterProblem->count())))); overlapMatrixEvaluator->disableMapCache();
	std::shared_ptr<RasterStripPackingCompactor> compactorMatrix = std::shared_ptr<RasterStripPackingCompactor>(new RasterStripPackingCompactor(args::get(argLength), rasterProblem, overlapMatrixEvaluator, 0.04, 0.01));
	start = std::chrono::system_clock::now();
	for (int i = 0; i < REPETITIONS; i++) {
		for (int k = 0; k < rasterProblem->count(); k++) {
			curMap = overlapMatrixEvaluator->getTotalOverlapMap(k, solution.getOrientation(k), solution);
		}
	}
	end = std::chrono::system_clock::now();
	std::cout << "Total elapsed time for " << REPETITIONS << " repetitions of matrix method was " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us." << std::endl;
	//curMap->getImage().save("map2.png");

	return 0;
}
