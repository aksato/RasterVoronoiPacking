#include "args/args.hxx"
#include "packingproblem.h"
#include "raster/rasterpackingproblem.h"
#include "raster/rasteroverlapevaluator.h"
#include "raster/rasterstrippackingparameters.h"
#include "raster/rasterstrippackingsolver.h"
#include "raster/rastersquarepackingcompactor.h"
#include <iostream>
#include <QString>
#include <QDir>
#include <QFileInfo>

#define REPETITIONS 1000

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
	std::shared_ptr<RASTERVORONOIPACKING::RasterPackingProblem> rasterProblem = std::shared_ptr<RASTERVORONOIPACKING::RasterPackingProblem>(new RASTERVORONOIPACKING::RasterPackingProblem(problem));
	RASTERVORONOIPACKING::RasterPackingSolution	solution = RASTERVORONOIPACKING::RasterPackingSolution(rasterProblem->count());
	QDir::setCurrent(originalPath);

	qsrand(4939495);
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solver = RASTERVORONOIPACKING::RasterStripPackingSolver::createRasterPackingSolver({ rasterProblem }, RASTERVORONOIPACKING::RasterStripPackingParameters(RASTERVORONOIPACKING::NONE, 1));
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingCompactor> compactor = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingCompactor>(new RASTERVORONOIPACKING::RasterStripPackingCompactor(args::get(argLength), rasterProblem, solver->getOverlapEvaluator(), 0.04, 0.01));

	auto start = std::chrono::system_clock::now();
	std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> curMap;
	for (int i = 0; i < REPETITIONS; i++) {
		compactor->generateRandomSolution(solution);
		for (int k = 0; k < rasterProblem->count(); k++) curMap = solver->overlapEvaluator->getTotalOverlapMap(k, solution.getOrientation(k), solution);
	}
	auto end = std::chrono::system_clock::now();
	std::cout << "Total elapsed time for " << REPETITIONS << " repetitions was " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us." << std::endl;

	curMap->getImage().save("map.png");
	solution.exportToPgf("sol.pgf", rasterProblem, (qreal)compactor->getCurrentLength() / rasterProblem->getScale(), (qreal)compactor->getCurrentHeight() / rasterProblem->getScale());

	return 0;
}
