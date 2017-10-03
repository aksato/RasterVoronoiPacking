#include <QCommandLineParser>
#include <QFile>
#include <QTextStream>
#include <QDebug>
#include "packingParametersParser.h"

CommandLineParseResult parseCommandLine(QCommandLineParser &parser, ConsolePackingArgs *params, QString *errorMessage)
{
    bool zoomedInputFileSet = false;
    parser.setSingleDashWordOptionMode(QCommandLineParser::ParseAsLongOptions);

    parser.addPositionalArgument("source","Input problem file path.");
    const QCommandLineOption nameZoomedInput(QStringList() << "zoom", "Zoomed problem input.", "name");
    parser.addOption(nameZoomedInput);
    const QCommandLineOption nameOutputTXT(QStringList() << "result", "The output result statistics file name.", "name");
    parser.addOption(nameOutputTXT);
    const QCommandLineOption nameOutputXML(QStringList() << "layout", "The output layout XML file name.", "name");
    parser.addOption(nameOutputXML);
	const QCommandLineOption boolOutputSeed("appendseed", "Automatically append seed value to output file names.");
	parser.addOption(boolOutputSeed);

    const QCommandLineOption typeMethod("method", "Raster packing method choices: default, gls, zoom, zoomgls.", "type");
    parser.addOption(typeMethod);
    const QCommandLineOption typeInitialSolution("initial", "Initial solution choices: random, bottomleft.", "type");
    parser.addOption(typeInitialSolution);
    const QCommandLineOption valueMaxWorseSolutions("nmo", "Maximum number of non-best solutions.", "value");
    parser.addOption(valueMaxWorseSolutions);
    const QCommandLineOption valueTimeLimit("duration", "Time limit in seconds.", "value");
    parser.addOption(valueTimeLimit);
	const QCommandLineOption valueIterationsLimit("maxits", "Maximum number of iterations.", "value");
	parser.addOption(valueIterationsLimit);
    const QCommandLineOption valueLenght("length", "Container lenght.", "value");
    parser.addOption(valueLenght);
	const QCommandLineOption boolStripPacking("strippacking", "Strip packing version.");
	parser.addOption(boolStripPacking);
	const QCommandLineOption valueNumThreads("parallel", "Number of parallel executions of the algorithm.", "value");
	parser.addOption(valueNumThreads);
	const QCommandLineOption placementMethod("placement", "Criteria for choosing positions when there are multiple minimum values (debug). Choices: bottomleft, random, limits and contour.", "type");
	parser.addOption(placementMethod);
	const QCommandLineOption valueCluster("clusterfactor", "Time fraction for cluster executuion.", "value");
	parser.addOption(valueCluster);
	const QCommandLineOption boolRectangularPacking("rectpacking", "Rectangular packing version.");
	parser.addOption(boolRectangularPacking);

    const QCommandLineOption helpOption = parser.addHelpOption();
    const QCommandLineOption versionOption = parser.addVersionOption();

    if (!parser.parse(QCoreApplication::arguments())) {
        *errorMessage = parser.errorText();
        return CommandLineError;
    }

    if (parser.isSet(versionOption))
        return CommandLineVersionRequested;

    if (parser.isSet(helpOption))
        return CommandLineHelpRequested;

    const QStringList positionalArguments = parser.positionalArguments();
    if (positionalArguments.isEmpty()) {
        *errorMessage = "Argument missing: 'source'.";
        return CommandLineError;
    }
    if (positionalArguments.size() > 1) {
        *errorMessage = "Too many arguments specified.";
        return CommandLineError;
    }
    params->inputFilePath = positionalArguments.at(0);

    if (parser.isSet(nameZoomedInput)) {
        const QString inputName = parser.value(nameZoomedInput);
        params->zoomedInputFilePath = inputName;
        zoomedInputFileSet = true;
    }

    if (parser.isSet(nameOutputTXT)) {
        const QString outputName = parser.value(nameOutputTXT);
        params->outputTXTFile = outputName;
    }
    else params->outputTXTFile = "outlog.dat";

    if (parser.isSet(nameOutputXML)) {
        const QString outputName = parser.value(nameOutputXML);
        params->outputXMLFile = outputName;
    }
    else params->outputXMLFile = "bestSol.xml";
	params->appendSeedToOutputFiles = parser.isSet(boolOutputSeed);

    if (parser.isSet(typeMethod)) {
        const QString methodType = parser.value(typeMethod).toLower();
        if(methodType != "default" && methodType != "gls" && methodType != "zoom" && methodType != "zoomgls") {
            *errorMessage = "Invalid method type! Avaible methods: 'default', 'gls', 'zoom' and 'zoomgls'.";
            return CommandLineError;
        }
        if(methodType == "default") params->methodType = Method_Default;
        if(methodType == "gls") params->methodType = Method_Gls;
        if(methodType == "zoom") params->methodType = Method_Zoom;
        if(methodType == "zoomgls") params->methodType = Method_ZoomGls;

        if(zoomedInputFileSet && (methodType == "default" || methodType == "gls"))
            qWarning() << "Warning: Method does not use zoom input file, ignoring the specified zoom problem.";
        if(!zoomedInputFileSet && methodType == "zoom") {
            qWarning() << "Warning: No zoom input file specified, changing method to 'default'";
            params->methodType = Method_Default;
        }
        if(!zoomedInputFileSet && methodType == "zoomgls") {
            qWarning() << "Warning: No zoom input file specified, changing method to 'gls'";
            params->methodType = Method_Gls;
        }
    }
    else {
        qWarning() << "Warning: Method not specified, set to default (no zoom, no gls).";
        params->methodType = Method_Default;
    }

    if (parser.isSet(typeInitialSolution)) {
        const QString solutionType = parser.value(typeInitialSolution).toLower();
		if (solutionType != "random" && solutionType != "bottomleft") {
            *errorMessage = "Invalid initial solution type! Avaible methods: 'random, bottomleft'.";
            return CommandLineError;
        }
        if(solutionType == "random") params->initialSolutionType = Solution_Random;
		if (solutionType == "bottomleft") params->initialSolutionType = Bottom_Left;
    }
    else params->initialSolutionType = Solution_Random;

    if (parser.isSet(valueMaxWorseSolutions)) {
        const QString nmoString = parser.value(valueMaxWorseSolutions);
        bool ok;
        const int nmo = nmoString.toInt(&ok);
        if(ok && nmo > 0) params->maxWorseSolutionsValue = nmo;
        else {
            *errorMessage = "Bad Nmo value.";
            return CommandLineError;
        }
    }
    else {
        qWarning() << "Warning: Nmo not found, set to default (200).";
        params->maxWorseSolutionsValue = 200;
    }

    if (parser.isSet(valueTimeLimit)) {
        const QString secsString = parser.value(valueTimeLimit);
        bool ok;
        const int timeLimit = secsString.toInt(&ok);
        if(ok && timeLimit > 0) params->timeLimitValue = timeLimit;
        else {
            *errorMessage = "Bad time limit value (must be expressed in seconds).";
            return CommandLineError;
        }
    }
    else {
        qWarning() << "Warning: Time limit not found, set to default (600s).";
        params->timeLimitValue = 600;
    }

	if (parser.isSet(valueIterationsLimit)) {
		const QString secsString = parser.value(valueIterationsLimit);
		bool ok;
		const int itLimit = secsString.toInt(&ok);
		if (ok && itLimit > 0) params->iterationsLimitValue = itLimit;
		else {
			*errorMessage = "Bad iteration limit value (must be expressed in seconds).";
			return CommandLineError;
		}
	}
	else {
		params->iterationsLimitValue = 0;
	}

    if (parser.isSet(valueLenght)) {
        const QString lengthString = parser.value(valueLenght);
        bool ok;
        const float containerLenght = lengthString.toFloat(&ok);
        if(ok && containerLenght > 0) {
            params->containerLenght = containerLenght;
            params->originalContainerLenght = false;
        }
        else {
            *errorMessage = "Bad container lenght value.";
            return CommandLineError;
        }
    }
    else params->originalContainerLenght = true;

	if (parser.isSet(boolStripPacking)) params->stripPacking = true;
	else  params->stripPacking = false;

	if (parser.isSet(boolRectangularPacking)) params->rectangularPacking = true;
	else  params->rectangularPacking = false;

	if (parser.isSet(valueNumThreads)) {
		const QString threadsString = parser.value(valueNumThreads);
		bool ok;
		const int nthreads = threadsString.toInt(&ok);
		if (ok && threadsString > 0) params->numThreads = nthreads;
		else {
			*errorMessage = "Bad parallel value.";
			return CommandLineError;
		}
	}
	else {
		params->numThreads = 1;
	}

	if (parser.isSet(placementMethod)) {
		const QString methodType = parser.value(placementMethod).toLower();
		if (methodType != "bottomleft" && methodType != "random" && methodType != "limits" && methodType != "contour") {
			*errorMessage = "Invalid method type! Avaible methods: 'bottomleft', 'random', 'limits' and 'contour'.";
			return CommandLineError;
		}
		if (methodType == "bottomleft") params->placementType = Pos_BottomLeft;
		if (methodType == "random") params->placementType = Pos_Random;
		if (methodType == "limits") params->placementType = Pos_Limits;
		if (methodType == "contour") params->placementType = Pos_Contour;
	}
	else params->placementType = Pos_BottomLeft;

	if (parser.isSet(valueCluster)) {
		const QString clusterString = parser.value(valueCluster);
		bool ok;
		const float clusterFactor = clusterString.toFloat(&ok);
		if (ok && clusterFactor >= 0 && clusterFactor <=1.0) {
			params->clusterFactor = clusterFactor;
		}
		else {
			*errorMessage = "Bad cluster factor value.";
			return CommandLineError;
		}
	}
	else params->clusterFactor = -1.0;

    return CommandLineOk;
}
