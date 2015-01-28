#include "rasterpackingsolution.h"
#include "rasterpackingproblem.h"
#include <QXmlStreamWriter>
#include <QFile>

using namespace RASTERVORONOIPACKING;

RasterPackingSolution::RasterPackingSolution()
{
}

RasterPackingSolution::RasterPackingSolution(int numItems)
{
    for(int i = 0; i < numItems; i++)
        placements.append(RasterItemPlacement());
}

QDebug operator<<(QDebug dbg, const RasterPackingSolution &c)
{
    for(int i = 0; i < c.getNumItems(); i++) {
        dbg.nospace() << "[Item:" << i << ", Pos:(" << c.getPosition(i).x() << "," << c.getPosition(i).y() << "), Angle:" << c.getOrientation(i) << "]\n";
    }

    return dbg.space();
}

bool RasterPackingSolution::save(QString fileName, std::shared_ptr<RasterPackingProblem> problem, qreal length, bool printSeed, uint seed) {
    QFile file(fileName);
    bool newFile = !file.exists();
    if(!file.open(QIODevice::Append)) {
        qCritical() << "Error: Cannot create output file" << fileName << ": " << qPrintable(file.errorString());
        return false;
    }

    QXmlStreamWriter stream;
    stream.setDevice(&file);
    stream.setAutoFormatting(true);
    if(newFile) stream.writeStartDocument();

    stream.writeStartElement("solution");
    for(int i = 0; i < placements.size(); i++) {
        stream.writeStartElement("placement");
        stream.writeAttribute("boardNumber", "1");
        stream.writeAttribute("x", QString::number(placements.at(i).getPos().x()/problem->getScale()));
        stream.writeAttribute("y", QString::number(placements.at(i).getPos().y()/problem->getScale()));
        stream.writeAttribute("idboard", problem->getContainerName());
        stream.writeAttribute("idPiece", problem->getItem(i)->getPieceName());
        stream.writeAttribute("angle", QString::number(problem->getItem(i)->getAngleValue(placements.at(i).getOrientation())));
        stream.writeAttribute("mirror", "none");
        stream.writeEndElement(); // placement
    }
    stream.writeStartElement("extraInfo");
    stream.writeTextElement("length", QString::number(length));
    stream.writeTextElement("scale", QString::number(problem->getScale()));
    if(printSeed) stream.writeTextElement("seed", QString::number(seed));
    stream.writeEndElement(); // extraInfo
    stream.writeEndElement(); // solution

    file.close();

    return true;
}
