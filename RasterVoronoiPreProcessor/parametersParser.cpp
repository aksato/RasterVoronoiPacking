#include <QCommandLineParser>
#include <QFile>
#include <QTextStream>
#include "parametersParser.h"

CommandLineParseResult parseCommandLine(QCommandLineParser &parser, PreProcessorParameters *params, QString *errorMessage)
{
    parser.setSingleDashWordOptionMode(QCommandLineParser::ParseAsLongOptions);
    parser.addPositionalArgument("source","Input problem file path.");
    parser.addPositionalArgument("destination","Destination directory.");
    const QCommandLineOption nameOutputXML(QStringList() << "output", "The output XML file name.", "outputName");
    parser.addOption(nameOutputXML);
    const QCommandLineOption valuePuzzleScale("nfp-scale", "Scale factor used in nofit calculations.", "puzzleScale");
    parser.addOption(valuePuzzleScale);
    const QCommandLineOption typePuzzle("problem-type", "Manual definition of input file type: esicup or cfrefp.", "problemType");
    parser.addOption(typePuzzle);
    const QCommandLineOption valueRasterScale("raster-scale", "Scale factor used in the rasterization process.", "rasterScale");
    parser.addOption(valueRasterScale);
	const QCommandLineOption valueFixScale("fix-scale", "Scale factor used to correct scaled CFREFP problems.", "fixScale");
    parser.addOption(valueFixScale);
    const QCommandLineOption boolSaveRaster("raster-output", "Save intermediate raster results.");
    parser.addOption(boolSaveRaster);
    const QCommandLineOption typeOutputFile("output-type", "Type of output file: 8bit-image or data.", "ouputType");
    parser.addOption(typeOutputFile);
    const QCommandLineOption nameHeaderFile("header-file", "Name of XML file to copy the header from.", "headerFileName");
    parser.addOption(nameHeaderFile);
    const QCommandLineOption nameOptionsFile("options-file", "Read options from a text file.", "fileName");
    parser.addOption(nameOptionsFile);
	const QCommandLineOption valueInnerFitEps("ifp-eps", "Epsilon used for inner-fit polygon rasterization.", "epsValue");
	parser.addOption(valueInnerFitEps);
	const QCommandLineOption boolNoOverlap("nooverlap", "Creates nofit polygons with contour (guarantees no overlap).");
	parser.addOption(boolNoOverlap);
	const QCommandLineOption nameClusterInfoFile("cluster-info", "XML formatted cluster information.", "fileName");
	parser.addOption(nameClusterInfoFile);;
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

    if (parser.isSet(nameOutputXML)) {
        const QString outputName = parser.value(nameOutputXML);
        params->outputXMLName = outputName;
    }
    else params->outputXMLName = "output.xml";

    if (parser.isSet(valuePuzzleScale)) {
        const QString puzzleScale = parser.value(valuePuzzleScale);
        bool ok;
        const float scale = puzzleScale.toFloat(&ok);
        if(ok && scale > 0) params->puzzleScaleFactor = scale;
        else {
            *errorMessage = "Bad nofit polygon scale value.";
            return CommandLineError;
        }
    }
    else params->puzzleScaleFactor = 1.0;

    if (parser.isSet(typePuzzle)) {
        const QString problemType = parser.value(typePuzzle).toLower();
        if(problemType != "esicup" && problemType != "cfrefp") {
            *errorMessage = "Problem type must be either 'esicup' or 'cfrefp'.";
            return CommandLineError;
        }
        else params->inputFileType = problemType;
    }
    else params->inputFileType = "";

    if (parser.isSet(valueRasterScale)) {
        const QString rasterScale = parser.value(valueRasterScale);
        bool ok;
        const float scale = rasterScale.toFloat(&ok);
        if(ok && scale > 0) params->rasterScaleFactor = scale;
        else {
            *errorMessage = "Bad raster scale value.";
            return CommandLineError;
        }
    }
    else params->rasterScaleFactor = 1.0;

	if (parser.isSet(valueFixScale)) {
        const QString fixScale = parser.value(valueFixScale);
        bool ok;
        const float scale = fixScale.toFloat(&ok);
		if(ok && scale > 0) params->scaleFixFactor = scale;
        else {
            *errorMessage = "Bad fix scale value.";
            return CommandLineError;
        }
    }
    else params->scaleFixFactor = 1.0;

    if (parser.isSet(boolSaveRaster)) {
        params->saveRaster = true;
    }
    else params->saveRaster = false;

    if (parser.isSet(typeOutputFile)) {
        const QString ouputType = parser.value(typeOutputFile).toLower();
        if(ouputType != "8bit-image" && ouputType != "data") {
            *errorMessage = "Output file type must be either '8bit-image' or 'data'.";
            return CommandLineError;
        }
        else params->outputFormat = ouputType;
    }
    else params->outputFormat = "8bit-image";

    if (parser.isSet(nameHeaderFile)) {
        const QString headerFile = parser.value(nameHeaderFile);
        params->headerFile = headerFile;
    }
    else params->headerFile = "";

    if (parser.isSet(nameOptionsFile)) {
        const QString optionsFile = parser.value(nameOptionsFile);
        params->optionsFile = optionsFile;
    }
    else params->headerFile = "";

	if (parser.isSet(valueInnerFitEps)) {
		const QString ifpEps = parser.value(valueInnerFitEps);
		bool ok;
		const float eps = ifpEps.toFloat(&ok);
		if (ok && eps > 0) params->innerFitEpsilon = eps;
		else {
			*errorMessage = "Bad epsilon value.";
			return CommandLineError;
		}
	}
	else params->innerFitEpsilon = -1.0;

	if (parser.isSet(boolNoOverlap)) params->noOverlap = true;
	else  params->noOverlap = false;

	if (parser.isSet(nameClusterInfoFile)) {
		const QString clusterInfoFile = parser.value(nameClusterInfoFile);
		params->clusterInfoFile = clusterInfoFile;
	}
	else params->clusterInfoFile = "";

    const QStringList positionalArguments = parser.positionalArguments();
    if (positionalArguments.isEmpty() || positionalArguments.size() == 1) {
        *errorMessage = "Arguments missing: 'source' or 'destination' or both.";
        return CommandLineError;
    }
    if (positionalArguments.size() > 2) {
        *errorMessage = "Several arguments specified.";
        return CommandLineError;
    }
    params->inputFilePath = positionalArguments.at(0);
    params->outputDir = positionalArguments.at(1);

    return CommandLineOk;
}

CommandLineParseResult parseOptionsFile(QString fileName, PreProcessorParameters *params, QString *errorMessage) {
    QFile file(fileName);
    if (!file.open(QFile::ReadOnly | QFile::Text)) {
        *errorMessage = "Cannot read options file.";
        return CommandLineError;
    }

    bool ok;
    QTextStream in(&file);
    while(!in.atEnd()) {
        QStringList line = in.readLine().split('=');
        if(line.size() != 2) {
            *errorMessage = "Syntax error while reading options file.";
            return CommandLineError;
        }

        if(line.at(0).toLower().trimmed() == "output") {
            const QString outputName = line.at(1).trimmed();
            params->outputXMLName = outputName;
        }

        if (line.at(0).toLower().trimmed() == "nfp-scale") {
            const QString puzzleScale = line.at(1).trimmed();
            const float scale = puzzleScale.toFloat(&ok);
            if(ok && scale > 0) params->puzzleScaleFactor = scale;
            else {
                *errorMessage = "Bad nofit polygon scale value.";
                return CommandLineError;
            }
        }

        if (line.at(0).toLower().trimmed() == "problem-type") {
            const QString problemType = line.at(1).toLower().trimmed();
            if(problemType != "esicup" && problemType != "cfrefp") {
                *errorMessage = "Problem type must be either 'esicup' or 'cfrefp'.";
                return CommandLineError;
            }
            else params->inputFileType = problemType;
        }

        if (line.at(0).toLower().trimmed() == "raster-scale") {
            const QString rasterScale = line.at(1);
            const float scale = rasterScale.toFloat(&ok);
            if(ok && scale > 0) params->rasterScaleFactor = scale;
            else {
                *errorMessage = "Bad raster scale value.";
                return CommandLineError;
            }
        }

		if (line.at(0).toLower().trimmed() == "fix-scale") {
            const QString fixScale = line.at(1);
            const float scale = fixScale.toFloat(&ok);
			if(ok && scale > 0) params->scaleFixFactor = scale;
            else {
                *errorMessage = "Bad fix scale value.";
                return CommandLineError;
            }
        }

        if (line.at(0).toLower().trimmed() == "raster-output") {
            const QString rasterOutput = line.at(1).toLower().trimmed();
            if(rasterOutput != "true" && rasterOutput != "false") {
                *errorMessage = "Raster output must be set to either 'true' or 'false'.";
                return CommandLineError;
            }
            else if(rasterOutput == "true") params->saveRaster = true;
            else params->saveRaster = false;
        }

        if (line.at(0).toLower().trimmed() == "output-type") {
            const QString ouputType = line.at(1).toLower().trimmed();
            if(ouputType != "8bit-image" && ouputType != "data") {
                *errorMessage = "Output file type must be either '8bit-image' or 'data'.";
                return CommandLineError;
            }
            else params->outputFormat = ouputType;
        }

        if (line.at(0).toLower().trimmed() == "header-file") {
            const QString headerFile = line.at(1).trimmed();
            params->headerFile = headerFile;
        }

		if (line.at(0).toLower().trimmed() == "ifp-eps") { // FIXME: Assign default value?
			const QString ifpEps = line.at(1);
			const float eps = ifpEps.toFloat(&ok);
			if (ok && eps > 0) params->innerFitEpsilon = eps;
			else {
				*errorMessage = "Bad epsilon value.";
				return CommandLineError;
			}
		}

		if (line.at(0).toLower().trimmed() == "nooverlap") { // FIXME: Assign default value?
			const QString isNoOverlap = line.at(1);
			const int yesorno = isNoOverlap.toInt(&ok);
			if (ok && yesorno == 0 || yesorno == 1) {
				if (yesorno == 0) params->noOverlap = false;
				else params->noOverlap = true;
			}
			else {
				*errorMessage = "Bad no overlap value.";
				return CommandLineError;
			}
		}

		if (line.at(0).toLower().trimmed() == "cluster-info") { // FIXME: Assign default value?
			const QString clusterInfoFile = line.at(1).trimmed();
			params->clusterInfoFile = clusterInfoFile;
		}
    }

    return CommandLineOk;
}
