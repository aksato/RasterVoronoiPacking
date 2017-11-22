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
    const QCommandLineOption valuePuzzleScale("nfps-cale", "Scale factor used in nofit calculations.", "puzzleScale");
    parser.addOption(valuePuzzleScale);
    const QCommandLineOption typePuzzle("problem-type", "Manual definition of input file type: esicup or cfrefp.", "problemType");
    parser.addOption(typePuzzle);
    const QCommandLineOption valueRasterScale("raster-scale", "Scale factor used in the rasterization process.", "rasterScale");
    parser.addOption(valueRasterScale);
	const QCommandLineOption valueFixScale("fix-scale", "Scale factor used to correct scaled CFREFP problems.", "fixScale");
    parser.addOption(valueFixScale);
    const QCommandLineOption boolSaveImage("image-output", "Save raster results in image format.");
	parser.addOption(boolSaveImage);
    const QCommandLineOption nameHeaderFile("header-file", "Name of XML file to copy the header from.", "headerFileName");
    parser.addOption(nameHeaderFile);
	const QCommandLineOption boolSkipDt("raster-only", "Skip distance transformation.");
	parser.addOption(boolSkipDt);
    const QCommandLineOption nameOptionsFile("options-file", "Read options from a text file.", "fileName");
    parser.addOption(nameOptionsFile);
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

	params->outputImages = parser.isSet(boolSaveImage);
	params->skipDt = parser.isSet(boolSkipDt);

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

        if (line.at(0).toLower().trimmed() == "image-output") {
            const QString imageOutputVal = line.at(1).toLower().trimmed();
			if (imageOutputVal != "true" && imageOutputVal != "false") {
                *errorMessage = "Raster output must be set to either 'true' or 'false'.";
                return CommandLineError;
            }
			else if (imageOutputVal == "true") params->outputImages = true;
			else params->outputImages = false;
        }

		if (line.at(0).toLower().trimmed() == "raster-only") {
			const QString rasterOnlyVal = line.at(1).toLower().trimmed();
			if (rasterOnlyVal != "true" && rasterOnlyVal != "false") {
				*errorMessage = "Raster output must be set to either 'true' or 'false'.";
				return CommandLineError;
			}
			else if (rasterOnlyVal == "true") params->skipDt = true;
			else params->skipDt = false;
		}

        if (line.at(0).toLower().trimmed() == "header-file") {
            const QString headerFile = line.at(1).trimmed();
            params->headerFile = headerFile;
        }
    }

    return CommandLineOk;
}
