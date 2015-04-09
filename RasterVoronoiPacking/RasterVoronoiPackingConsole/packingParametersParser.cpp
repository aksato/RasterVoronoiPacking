#include <QCommandLineParser>
#include <QFile>
#include <QTextStream>
#include <QDebug>
#include "packingParametersParser.h"

CommandLineParseResult parseCommandLine(QCommandLineParser &parser, PackingParameters *params, QString *errorMessage)
{
    bool zoomedInputFileSet = false;
    parser.setSingleDashWordOptionMode(QCommandLineParser::ParseAsLongOptions);

    parser.addPositionalArgument("source","Input problem file path.");
    const QCommandLineOption nameZoomedInput(QStringList() << "zoom-problem", "Zoomed problem input.", "name");
    parser.addOption(nameZoomedInput);
    const QCommandLineOption nameOutputTXT(QStringList() << "result", "The output result statistics file name.", "name");
    parser.addOption(nameOutputTXT);
    const QCommandLineOption nameOutputXML(QStringList() << "layout", "The output layout XML file name.", "name");
    parser.addOption(nameOutputXML);

    const QCommandLineOption typeMethod("method", "Raster packing method choices: default, gls, zoom, zoomgls.", "type");
    parser.addOption(typeMethod);
    const QCommandLineOption typeInitialSolution("initial-solution", "Initial solution choices: random.", "type");
    parser.addOption(typeInitialSolution);
    const QCommandLineOption valueMaxWorseSolutions("nmo", "Maximum number of non-best solutions.", "value");
    parser.addOption(valueMaxWorseSolutions);
    const QCommandLineOption valueTimeLimit("duration", "Time limit in seconds.", "value");
    parser.addOption(valueTimeLimit);
    const QCommandLineOption valueLenght("length", "Container lenght.", "value");
    parser.addOption(valueLenght);
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
        if(solutionType != "random") {
            *errorMessage = "Invalid initial solution type! Avaible methods: 'random'.";
            return CommandLineError;
        }
        if(solutionType == "random") params->initialSolutionType = Solution_Random;
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

    return CommandLineOk;
}