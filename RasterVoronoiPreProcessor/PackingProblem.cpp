#include "PackingProblem.h"
#include <QString>
#include <QStringList>
#include <QFile>
#include <QXmlStreamReader>
#include <QDebug>
#include <QMap>
#include <QFileInfo>
#include "polybool/polybool.h"
#include "annealing/cPolygon.h"
#include "annealing/cShape.h"
#include "annealing/cContainer.h"
#include "annealing/cRectangularContainer.h"
#include "annealing/cShapeInlines.h"
#include "annealing/problemInstance.h"

using namespace RASTERPACKING;

bool IsAtLeft(QPointF &v1, QPointF &v2)
{
    qreal res = v1.x()*v2.y()-v2.x()*v1.y();
    if(qFuzzyCompare ( 1 + 0.0, 1 + res ))
        // paralel vectors
        return (v1.x()*v2.x() + v1.y()*v1.y() < 0);
    if(res>0) return false;
    return true;
}

std::shared_ptr<Polygon> Polygon::getNofitPolygon(std::shared_ptr<Polygon> staticPolygon, std::shared_ptr<Polygon> orbitingPolygon) {
    QPointF v1, nv1, v2, nv2, dv1, dv2;

    int v1Index, nv1Index;
    int v2Index, nv2Index;
    int sCount, oCount;

    std::shared_ptr<Polygon> nofitResult = std::shared_ptr<Polygon>(new Polygon);
    sCount = staticPolygon->size();
    oCount = orbitingPolygon->size();
    // Take the first vertex on Obstacle
    v2 = staticPolygon->at(v2Index = 0);

    // We retrieve the incident edge on v2
    nv2 = staticPolygon->at(nv2Index = (v2Index+1) % sCount);
    dv2 = nv2 - v2;

    // Now, we try to find the apropriate vertex on the object
    //	to fit. The apropriate vertex is the source of first
    //	edge at left of the current edge on obstacle
    v1 = orbitingPolygon->at(v1Index = 0);

    nv1 = orbitingPolygon->at(nv1Index = (v1Index+1) % oCount);
    // Remember: All coordinates on ToFit INVERTED
    dv1 = v1 - nv1;

    // Edge on ToFit is at right, forward til first edge
    //  at right is found
    if(!IsAtLeft(dv1, dv2)) {
        do {
            v1 = nv1; v1Index = nv1Index;
            nv1 = orbitingPolygon->at(nv1Index = (nv1Index+1) % oCount);
            dv1 = v1 - nv1;
        } while(!IsAtLeft(dv1, dv2));
    }
    // Edge is at left. However, we don't know for
    //  sure if it is the last edge at right...
    //	backward til an edge at right is found, then return
    else {
        do {
            nv1 = v1; nv1Index = v1Index;
            if(v1Index == 0) v1Index = oCount;
            v1 = orbitingPolygon->at(v1Index = (v1Index-1) % oCount);
            //            v1 = orbitingPolygon->at(v1Index = (v1Index-1) % oCount);
            dv1 = v1 - nv1;
        } while(IsAtLeft(dv1, dv2));
        // then return one position...
        v1 = nv1; v1Index = nv1Index;
        nv1 = orbitingPolygon->at(nv1Index = (nv1Index+1) % oCount);
        dv1 = v1 - nv1;
    }

    // Origin found: Add the vertex to the dest polygon
    nofitResult->push_back(v2 - v1);

    // Get first edge from obstacle polygon
    v2 = nv2; v2Index = nv2Index;
    nv2 = staticPolygon->at(nv2Index = (nv2Index+1) % sCount);
    dv2 = nv2 - v2;

    nofitResult->push_back(v2 - v1);

    int n = oCount + sCount;
    for(int i=2;i<n;i++) {
        // Get edge from Obstacle polygon
        if(IsAtLeft(dv1, dv2)) {
            v2 = nv2; v2Index = nv2Index;
            nv2 = staticPolygon->at(nv2Index = (nv2Index+1) % sCount);
            dv2 = nv2 - v2;
        }
        // Get edge from Obstacle polygon
        else {
            v1 = nv1; v1Index = nv1Index;
            nv1 = orbitingPolygon->at(nv1Index = (nv1Index+1) % oCount);
            dv1 = v1 - nv1;
        }
        nofitResult->push_back(v2 - v1);
    }

    return nofitResult;
}

void Bresenham(int x1, int y1, int const x2, int const y2, QImage &image) {
    int delta_x(x2 - x1);
    // if x1 == x2, then it does not matter what we set here
    signed char const ix((delta_x > 0) - (delta_x < 0));
    delta_x = std::abs(delta_x) << 1;

    int delta_y(y2 - y1);
    // if y1 == y2, then it does not matter what we set here
    signed char const iy((delta_y > 0) - (delta_y < 0));
    delta_y = std::abs(delta_y) << 1;

    image.setPixel(x1, y1, 0);

    if (delta_x >= delta_y)
    {
        // error may go below zero
        int error(delta_y - (delta_x >> 1));

        while (x1 != x2)
        {
            if ((error >= 0) && (error || (ix > 0)))
            {
                error -= delta_x;
                y1 += iy;
            }
            // else do nothing

            error += delta_y;
            x1 += ix;

            image.setPixel(x1, y1, 0);
        }
    }
    else
    {
        // error may go below zero
        int error(delta_x - (delta_y >> 1));

        while (y1 != y2)
        {
            if ((error >= 0) && (error || (iy > 0)))
            {
                error -= delta_y;
                x1 += ix;
            }
            // else do nothing

            error += delta_x;
            y1 += iy;

            image.setPixel(x1, y1, 0);
        }
    }
}

void BresenhamVec(int x1, int y1, int const x2, int const y2, int *S, int width) {
    int delta_x(x2 - x1);
    // if x1 == x2, then it does not matter what we set here
    signed char const ix((delta_x > 0) - (delta_x < 0));
    delta_x = std::abs(delta_x) << 1;

    int delta_y(y2 - y1);
    // if y1 == y2, then it does not matter what we set here
    signed char const iy((delta_y > 0) - (delta_y < 0));
    delta_y = std::abs(delta_y) << 1;

    S[y1*width+x1] = 1;

    if (delta_x >= delta_y)
    {
        // error may go below zero
        int error(delta_y - (delta_x >> 1));

        while (x1 != x2)
        {
            if ((error >= 0) && (error || (ix > 0)))
            {
                error -= delta_x;
                y1 += iy;
            }
            // else do nothing

            error += delta_y;
            x1 += ix;

            S[y1*width+x1] = 1;
        }
    }
    else
    {
        // error may go below zero
        int error(delta_x - (delta_y >> 1));

        while (y1 != y2)
        {
            if ((error >= 0) && (error || (iy > 0)))
            {
                error -= delta_y;
                x1 += ix;
            }
            // else do nothing

            error += delta_x;
            y1 += iy;

            S[y1*width+x1] = 1;
        }
    }
}

int *Polygon::getRasterImageVector(QPoint &RP, qreal scale, int &width, int &height) {
    QPolygonF polygon;
    qreal xMin, xMax, yMin, yMax;

    const_iterator it = cbegin();
    xMin = scale*(*it).x(); xMax = scale*(*it).x();
    yMin = scale*(*it).y(); yMax = scale*(*it).y();
    for(int i = 0; i < size(); i++) {
        qreal x = scale*at(i).x();
        qreal y = scale*at(i).y();

        if(x < xMin) xMin = x;
        if(x > xMax) xMax = x;
        if(y < yMin) yMin = y;
        if(y > yMax) yMax = y;

        polygon << QPointF(x, y);
    }
	RP.setX(-qRound(xMin)); RP.setY(-qRound(yMin));
    polygon.translate(-xMin,-yMin);

    width = qRound(xMax - xMin) + 1;
    height = qRound(yMax - yMin) + 1;
    int *S = new int[width*height];
    std::fill_n(S, width*height, 1);

    int curScanline = width;
    for (int pixelY=1; pixelY < height-1; pixelY++) {

        //  Build a list of nodes.
        QVector<int> nodeX;
        int j = polygon.size()-1;
        for(int i = 0; i < polygon.size(); i++) {
            qreal polyYi = polygon[i].y();
            qreal polyXi = polygon[i].x();
            qreal polyYj = polygon[j].y();
            qreal polyXj = polygon[j].x();

            if ( (polyYi<(double)pixelY && polyYj>=(double)pixelY) ||  (polyYj<(double)pixelY && polyYi>=(double)pixelY)) {
              nodeX.push_back((int)(polyXi+(pixelY-polyYi)/(polyYj-polyYi)*(polyXj-polyXi)));
            }
            j=i;
        }

        // Sort
        qSort(nodeX);


        //  Fill the pixels between node pairs.
        for(int i=0; i<nodeX.size(); i+=2) {
            int line = curScanline;
            line += nodeX[i]+1;
            for (j=nodeX[i]+1; j < nodeX[i+1]; j++, line+=1)
                S[line] = 0;
        }

        curScanline += width;
    }


    // FIXME: Delete horizontal lines
    QPolygonF::const_iterator itj;
    for(QPolygonF::const_iterator iti = polygon.cbegin(); iti != polygon.cend(); iti++) {
        if(iti+1 == polygon.cend()) itj = polygon.cbegin();
        else itj = iti+1;
        if((int)(*iti).y() == (int)(*itj).y()) {
            int left = (*iti).x() < (*itj).x() ? (*iti).x() : (*itj).x();
            int right = (*iti).x() < (*itj).x() ? (*itj).x() : (*iti).x();
            for(int i = left; i <= right; i++) {
                S[(int)(*iti).y()*width+i] = 1;
            }
        }
    }

    // FIXME: Test
    for(QVector<QPair<QPointF,QPointF>>::const_iterator it = this->degLines.cbegin(); it != this->degLines.cend(); it++) {
        QPointF p1 = scale*(*it).first; p1 -= QPointF(xMin, yMin);
        QPointF p2 = scale*(*it).second; p2 -= QPointF(xMin, yMin);
        BresenhamVec(qRound(p1.x()), qRound(p1.y()), qRound(p2.x()), qRound(p2.y()), S, width);
    }
    for(QVector<QPointF>::const_iterator it = this->degNodes.cbegin(); it != this->degNodes.cend(); it++) {
        QPointF p1 = scale*(*it); p1 -= QPointF(xMin, yMin);
         S[(int)p1.y()*width+(int)p1.x()] = 1;
    }

    return S;
}

QImage Polygon::getRasterImage8bit(QPoint &RP, qreal scale) {
    int width, heigth;
    QPolygonF polygon;
    qreal xMin, xMax, yMin, yMax;

    const_iterator it = cbegin();
    xMin = scale*(*it).x(); xMax = scale*(*it).x();
    yMin = scale*(*it).y(); yMax = scale*(*it).y();
    for(int i = 0; i < size(); i++) {
        qreal x = scale*at(i).x();
        qreal y = scale*at(i).y();

        if(x < xMin) xMin = x;
        if(x > xMax) xMax = x;
        if(y < yMin) yMin = y;
        if(y > yMax) yMax = y;

        polygon << QPointF(x, y);
    }
    RP.setX(-qRound(xMin)); RP.setY(-qRound(yMin));
    polygon.translate(-xMin,-yMin);

    width = qRound(xMax - xMin) + 1;
    heigth = qRound(yMax - yMin) + 1;
    QImage image(width, heigth, QImage::Format_Indexed8);
    image.setColor(0, qRgb(255, 255, 255));
    image.setColor(1, qRgb(255, 0, 0));
    image.fill(0);


    for (int pixelY=1; pixelY < image.height()-1; pixelY++) {
        //  Build a list of nodes.
        QVector<int> nodeX;
        int j = polygon.size()-1;
        for(int i = 0; i < polygon.size(); i++) {
            qreal polyYi = polygon[i].y();
            qreal polyXi = polygon[i].x();
            qreal polyYj = polygon[j].y();
            qreal polyXj = polygon[j].x();

            if ( (polyYi<(double)pixelY && polyYj>=(double)pixelY) ||  (polyYj<(double)pixelY && polyYi>=(double)pixelY)) {
              nodeX.push_back((int)(polyXi+(pixelY-polyYi)/(polyYj-polyYi)*(polyXj-polyXi)));
            }
            j=i;
        }

        // Sort
        qSort(nodeX);


        //  Fill the pixels between node pairs.
        for(int i=0; i<nodeX.size(); i+=2) {
            uchar *line = (uchar *)image.scanLine(pixelY);
            line += nodeX[i]+1;
            for (j=nodeX[i]+1; j < nodeX[i+1]; j++, line+=1)
                *line = 1;
        }

//        for(int i=0; i<nodeX.size(); i+=2) {

//            // FIXME: Special Case verification necessary?
//            if(nodeX[i]+1 > nodeX[i+1]-1) continue;

//            uchar *line = (uchar *)image.scanLine(pixelY);
//            int curLineIndex = 0;

//            int initialCurLineIndex = (nodeX[i]+1) / 8;
//            int initialCurLineOffset = (nodeX[i]+1) % 8;
//            int finalCurLineIndex = (nodeX[i+1]-1) / 8;
//            int finalCurLineOffset = (nodeX[i+1]-1) % 8;

//            line += (curLineIndex = initialCurLineIndex);
//            if(initialCurLineIndex != finalCurLineIndex) *line = *line | 255 >> initialCurLineOffset;
//            else {
//                *line = *line | ( (uchar)(255 >> initialCurLineOffset) & (uchar)(255 << (7-finalCurLineOffset)) );
//                continue;
//            }

//            curLineIndex++;
//            line++;
//            while(curLineIndex < finalCurLineIndex) {
//                *line = 255;
//                curLineIndex++;
//                line++;
//            }

//            *line = (uchar)(255 << (7-finalCurLineOffset));
//        }
    }


    // FIXME: Delete horizontal lines
    QPolygonF::const_iterator itj;
    for(QPolygonF::const_iterator iti = polygon.cbegin(); iti != polygon.cend(); iti++) {
        if(iti+1 == polygon.cend()) itj = polygon.cbegin();
        else itj = iti+1;
        if((int)(*iti).y() == (int)(*itj).y()) {
            int left = (*iti).x() < (*itj).x() ? (*iti).x() : (*itj).x();
            int right = (*iti).x() < (*itj).x() ? (*itj).x() : (*iti).x();
            for(int i = left; i <= right; i++) {
                image.setPixel(i,(int)(*iti).y(),0);
            }
        }
    }

    // FIXME: Test
    for(QVector<QPair<QPointF,QPointF>>::const_iterator it = this->degLines.cbegin(); it != this->degLines.cend(); it++) {
        QPointF p1 = scale*(*it).first; p1 -= QPointF(xMin, yMin);
        QPointF p2 = scale*(*it).second; p2 -= QPointF(xMin, yMin);
        Bresenham(qRound(p1.x()), qRound(p1.y()), qRound(p2.x()), qRound(p2.y()), image);
    }
    for(QVector<QPointF>::const_iterator it = this->degNodes.cbegin(); it != this->degNodes.cend(); it++) {
        QPointF p1 = scale*(*it); p1 -= QPointF(xMin, yMin);
        image.setPixel(p1.x(), p1.y(), 0);
    }

    return image;
}

QImage Polygon::getRasterImage(QPoint &RP, qreal scale) {
    int width, heigth;
    QPolygonF polygon;
    qreal xMin, xMax, yMin, yMax;

    const_iterator it = cbegin();
    xMin = scale*(*it).x(); xMax = scale*(*it).x();
    yMin = scale*(*it).y(); yMax = scale*(*it).y();
    for(int i = 0; i < size(); i++) {
        qreal x = scale*at(i).x();
        qreal y = scale*at(i).y();

        if(x < xMin) xMin = x;
        if(x > xMax) xMax = x;
        if(y < yMin) yMin = y;
        if(y > yMax) yMax = y;

        polygon << QPointF(x, y);
    }
    RP.setX(-qRound(xMin)); RP.setY(-qRound(yMin));
    polygon.translate(-xMin,-yMin);

    width = qRound(xMax - xMin) + 1;
    heigth = qRound(yMax - yMin) + 1;
    QImage image(width, heigth, QImage::Format_Mono);
    image.setColor(0, qRgb(255, 255, 255));
    image.setColor(1, qRgb(255, 0, 0));
    image.fill(0);


    for (int pixelY=1; pixelY < image.height()-1; pixelY++) {
        //  Build a list of nodes.
        QVector<int> nodeX;
        int j = polygon.size()-1;
        for(int i = 0; i < polygon.size(); i++) {
            qreal polyYi = polygon[i].y();
            qreal polyXi = polygon[i].x();
            qreal polyYj = polygon[j].y();
            qreal polyXj = polygon[j].x();

            if ( (polyYi<(double)pixelY && polyYj>=(double)pixelY) ||  (polyYj<(double)pixelY && polyYi>=(double)pixelY)) {
              nodeX.push_back((int)(polyXi+(pixelY-polyYi)/(polyYj-polyYi)*(polyXj-polyXi)));
            }
            j=i;
        }

        // Sort
        qSort(nodeX);

        for(int i=0; i<nodeX.size(); i+=2) {

            // FIXME: Special Case verification necessary?
            if(nodeX[i]+1 > nodeX[i+1]-1) continue;

            uchar *line = (uchar *)image.scanLine(pixelY);
            int curLineIndex = 0;

            int initialCurLineIndex = (nodeX[i]+1) / 8;
            int initialCurLineOffset = (nodeX[i]+1) % 8;
            int finalCurLineIndex = (nodeX[i+1]-1) / 8;
            int finalCurLineOffset = (nodeX[i+1]-1) % 8;

            line += (curLineIndex = initialCurLineIndex);
            if(initialCurLineIndex != finalCurLineIndex) *line = *line | 255 >> initialCurLineOffset;
            else {
                *line = *line | ( (uchar)(255 >> initialCurLineOffset) & (uchar)(255 << (7-finalCurLineOffset)) );
                continue;
            }

            curLineIndex++;
            line++;
            while(curLineIndex < finalCurLineIndex) {
                *line = 255;
                curLineIndex++;
                line++;
            }

            *line = (uchar)(255 << (7-finalCurLineOffset));
        }
    }


    // FIXME: Delete horizontal lines
    QPolygonF::const_iterator itj;
    for(QPolygonF::const_iterator iti = polygon.cbegin(); iti != polygon.cend(); iti++) {
        if(iti+1 == polygon.cend()) itj = polygon.cbegin();
        else itj = iti+1;
        if((int)(*iti).y() == (int)(*itj).y()) {
            int left = (*iti).x() < (*itj).x() ? (*iti).x() : (*itj).x();
            int right = (*iti).x() < (*itj).x() ? (*itj).x() : (*iti).x();
            for(int i = left; i <= right; i++) {
                image.setPixel(i,(int)(*iti).y(),0);
            }
        }
    }

    // FIXME: Test
    for(QVector<QPair<QPointF,QPointF>>::const_iterator it = this->degLines.cbegin(); it != this->degLines.cend(); it++) {
        QPointF p1 = scale*(*it).first; p1 -= QPointF(xMin, yMin);
        QPointF p2 = scale*(*it).second; p2 -= QPointF(xMin, yMin);
        Bresenham(qRound(p1.x()), qRound(p1.y()), qRound(p2.x()), qRound(p2.y()), image);
    }
    for(QVector<QPointF>::const_iterator it = this->degNodes.cbegin(); it != this->degNodes.cend(); it++) {
        QPointF p1 = scale*(*it); p1 -= QPointF(xMin, yMin);
        image.setPixel(p1.x(), p1.y(), 0);
    }

    return image;
}

QStringList Polygon::getXML() {
    QStringList commands;

    commands.push_back("writeStartElement"); commands.push_back("polygon");
    commands.push_back("writeAttribute"); commands.push_back("id"); commands.push_back(this->name);
    commands.push_back("writeAttribute"); commands.push_back("nVertices"); commands.push_back(QString::number(this->size()));
    commands.push_back("writeStartElement"); commands.push_back("lines");

    qreal xMin, xMax, yMin, yMax;
    const_iterator it = cbegin();
    xMin = (*it).x(); xMax = (*it).x();
    yMin = (*it).y(); yMax = (*it).y();

    for(int i = 0; i < size(); i++) {
        qreal x0, x1, y0, y1;
        x0 = at(i).x();
        y0 = at(i).y();

        if(i+1 == size()) {
            x1 = at(0).x();
            y1 = at(0).y();
        }
        else {
            x1 = at(i+1).x();
            y1 = at(i+1).y();
        }

        if(x0 < xMin) xMin = x0;
        if(x0 > xMax) xMax = x0;
        if(y0 < yMin) yMin = y0;
        if(y0 > yMax) yMax = y0;

        commands.push_back("writeStartElement"); commands.push_back("segment");
        commands.push_back("writeAttribute"); commands.push_back("n"); commands.push_back(QString::number(i+1));
        commands.push_back("writeAttribute"); commands.push_back("x0"); commands.push_back(QString::number(x0));
        commands.push_back("writeAttribute"); commands.push_back("x1"); commands.push_back(QString::number(x1));
        commands.push_back("writeAttribute"); commands.push_back("y0"); commands.push_back(QString::number(y0));
        commands.push_back("writeAttribute"); commands.push_back("y1"); commands.push_back(QString::number(y1));
        commands.push_back("writeEndElement"); // segment
    }

    commands.push_back("writeTextElement"); commands.push_back("xMin"); commands.push_back(QString::number(xMin));
    commands.push_back("writeTextElement"); commands.push_back("xMax"); commands.push_back(QString::number(xMax));
    commands.push_back("writeTextElement"); commands.push_back("yMin"); commands.push_back(QString::number(yMin));
    commands.push_back("writeTextElement"); commands.push_back("yMax"); commands.push_back(QString::number(yMax));
    commands.push_back("writeEndElement"); // lines
    commands.push_back("writeEndElement"); // polygon

    return commands;
}

QStringList Container::getXML() {
    QStringList commands;

    commands.push_back("writeStartElement"); commands.push_back("piece");
    commands.push_back("writeAttribute"); commands.push_back("id"); commands.push_back(this->name);
    commands.push_back("writeAttribute"); commands.push_back("quantity"); commands.push_back(QString::number(this->multiplicity));
    commands.push_back("writeStartElement"); commands.push_back("component");
    commands.push_back("writeAttribute"); commands.push_back("idPolygon"); commands.push_back(this->getPolygon()->getName());
    commands.push_back("writeAttribute"); commands.push_back("type"); commands.push_back("0");
    commands.push_back("writeAttribute"); commands.push_back("xOffset"); commands.push_back("0");
    commands.push_back("writeAttribute"); commands.push_back("yOffset"); commands.push_back("0");
    commands.push_back("writeEndElement"); // component
    commands.push_back("writeEndElement"); // piece

    return commands;
}

QStringList Piece::getXML() {
    QStringList commands;

    commands.push_back("writeStartElement"); commands.push_back("piece");
    commands.push_back("writeAttribute"); commands.push_back("id"); commands.push_back(this->name);
    commands.push_back("writeAttribute"); commands.push_back("quantity"); commands.push_back(QString::number(this->multiplicity));

    commands.push_back("writeStartElement"); commands.push_back("orientation");
    if(this->getOrientationsCount() == 0) {
        commands.push_back("writeStartElement"); commands.push_back("enumeration");
        commands.push_back("writeAttribute"); commands.push_back("angle"); commands.push_back("0");
        commands.push_back("writeEndElement"); // enumeration
    }
    else {
        for(QVector<unsigned int>::const_iterator it = this->corbegin(); it != this->corend(); it++) {
            commands.push_back("writeStartElement"); commands.push_back("enumeration");
            commands.push_back("writeAttribute"); commands.push_back("angle"); commands.push_back(QString::number(*it));
            commands.push_back("writeEndElement"); // enumeration
        }
    }
    commands.push_back("writeEndElement"); // orientation

    commands.push_back("writeStartElement"); commands.push_back("component");
    commands.push_back("writeAttribute"); commands.push_back("idPolygon"); commands.push_back(this->getPolygon()->getName());
    commands.push_back("writeAttribute"); commands.push_back("type"); commands.push_back("0");
    commands.push_back("writeAttribute"); commands.push_back("xOffset"); commands.push_back("0");
    commands.push_back("writeAttribute"); commands.push_back("yOffset"); commands.push_back("0");
    commands.push_back("writeEndElement"); // component
    commands.push_back("writeEndElement"); // piece

    return commands;
}

QStringList NoFitPolygon::getXML() {
    QStringList commands;

    commands.push_back("writeStartElement"); commands.push_back("nfp");
    commands.push_back("writeStartElement"); commands.push_back("staticPolygon");
    commands.push_back("writeAttribute"); commands.push_back("angle"); commands.push_back(QString::number(this->staticAngle));
    commands.push_back("writeAttribute"); commands.push_back("idPolygon"); commands.push_back(this->staticName);
    commands.push_back("writeAttribute"); commands.push_back("mirror"); commands.push_back("none");
    commands.push_back("writeEndElement"); // staticPolygon
    commands.push_back("writeStartElement"); commands.push_back("orbitingPolygon");
    commands.push_back("writeAttribute"); commands.push_back("angle"); commands.push_back(QString::number(this->orbitingAngle));
    commands.push_back("writeAttribute"); commands.push_back("idPolygon"); commands.push_back(this->orbitingName);
    commands.push_back("writeAttribute"); commands.push_back("mirror"); commands.push_back("none");
    commands.push_back("writeEndElement"); // orbitingPolygon
    commands.push_back("writeStartElement"); commands.push_back("resultingPolygon");
    commands.push_back("writeAttribute"); commands.push_back("idPolygon"); commands.push_back(this->name);
    commands.push_back("writeEndElement"); // resultingPolygon
    commands.push_back("writeEndElement"); // nfp

    return commands;
}

QStringList InnerFitPolygon::getXML() {
    QStringList commands;

    commands.push_back("writeStartElement"); commands.push_back("ifp");
    commands.push_back("writeStartElement"); commands.push_back("staticPolygon");
    commands.push_back("writeAttribute"); commands.push_back("angle"); commands.push_back(QString::number(this->staticAngle));
    commands.push_back("writeAttribute"); commands.push_back("idPolygon"); commands.push_back(this->staticName);
    commands.push_back("writeAttribute"); commands.push_back("mirror"); commands.push_back("none");
    commands.push_back("writeEndElement"); // staticPolygon
    commands.push_back("writeStartElement"); commands.push_back("orbitingPolygon");
    commands.push_back("writeAttribute"); commands.push_back("angle"); commands.push_back(QString::number(this->orbitingAngle));
    commands.push_back("writeAttribute"); commands.push_back("idPolygon"); commands.push_back(this->orbitingName);
    commands.push_back("writeAttribute"); commands.push_back("mirror"); commands.push_back("none");
    commands.push_back("writeEndElement"); // orbitingPolygon
    commands.push_back("writeStartElement"); commands.push_back("resultingPolygon");
    commands.push_back("writeAttribute"); commands.push_back("idPolygon"); commands.push_back(this->name);
    commands.push_back("writeEndElement"); // resultingPolygon
    commands.push_back("writeEndElement"); // ifp

    return commands;
}

QStringList RasterNoFitPolygon::getXML() {
    QStringList commands;

    commands.push_back("writeStartElement"); commands.push_back("rnfp");
    commands.push_back("writeStartElement"); commands.push_back("staticPolygon");
    commands.push_back("writeAttribute"); commands.push_back("angle"); commands.push_back(QString::number(this->staticAngle));
    commands.push_back("writeAttribute"); commands.push_back("idPolygon"); commands.push_back(this->staticName);
    commands.push_back("writeAttribute"); commands.push_back("mirror"); commands.push_back("none");
    commands.push_back("writeEndElement"); // staticPolygon
    commands.push_back("writeStartElement"); commands.push_back("orbitingPolygon");
    commands.push_back("writeAttribute"); commands.push_back("angle"); commands.push_back(QString::number(this->orbitingAngle));
    commands.push_back("writeAttribute"); commands.push_back("idPolygon"); commands.push_back(this->orbitingName);
    commands.push_back("writeAttribute"); commands.push_back("mirror"); commands.push_back("none");
    commands.push_back("writeEndElement"); // orbitingPolygon
    commands.push_back("writeStartElement"); commands.push_back("resultingImage");
    commands.push_back("writeAttribute"); commands.push_back("path"); commands.push_back(this->fileName);
    commands.push_back("writeAttribute"); commands.push_back("scale"); commands.push_back(QString::number(this->scale));
    commands.push_back("writeAttribute"); commands.push_back("x0"); commands.push_back(QString::number(this->referencePoint.x()));
    commands.push_back("writeAttribute"); commands.push_back("y0"); commands.push_back(QString::number(this->referencePoint.y()));
	commands.push_back("writeAttribute"); commands.push_back("maxD"); commands.push_back(QString::number(this->maxD));
    commands.push_back("writeEndElement"); // resultingPolygon
    commands.push_back("writeEndElement"); // rnfp

    return commands;
}

QStringList RasterInnerFitPolygon::getXML() {
    QStringList commands;

    commands.push_back("writeStartElement"); commands.push_back("rifp");
    commands.push_back("writeStartElement"); commands.push_back("staticPolygon");
    commands.push_back("writeAttribute"); commands.push_back("angle"); commands.push_back(QString::number(this->staticAngle));
    commands.push_back("writeAttribute"); commands.push_back("idPolygon"); commands.push_back(this->staticName);
    commands.push_back("writeAttribute"); commands.push_back("mirror"); commands.push_back("none");
    commands.push_back("writeEndElement"); // staticPolygon
    commands.push_back("writeStartElement"); commands.push_back("orbitingPolygon");
    commands.push_back("writeAttribute"); commands.push_back("angle"); commands.push_back(QString::number(this->orbitingAngle));
    commands.push_back("writeAttribute"); commands.push_back("idPolygon"); commands.push_back(this->orbitingName);
    commands.push_back("writeAttribute"); commands.push_back("mirror"); commands.push_back("none");
    commands.push_back("writeEndElement"); // orbitingPolygon
    commands.push_back("writeStartElement"); commands.push_back("resultingImage");
    commands.push_back("writeAttribute"); commands.push_back("path"); commands.push_back(this->fileName);
    commands.push_back("writeAttribute"); commands.push_back("scale"); commands.push_back(QString::number(this->scale));
    commands.push_back("writeAttribute"); commands.push_back("x0"); commands.push_back(QString::number(this->referencePoint.x()));
    commands.push_back("writeAttribute"); commands.push_back("y0"); commands.push_back(QString::number(this->referencePoint.y()));
    commands.push_back("writeEndElement"); // resultingPolygon
    commands.push_back("writeEndElement"); // rnfp

    return commands;
}

enum ReadStates {NEUTRAL, BOARDS_READING, LOT_READING};

Container::Container(QString _name, int _multiplicity) {
    this->name = _name;
    this->multiplicity = _multiplicity;
}

Piece::Piece(QString _name, int _multiplicity) {
    this->name = _name;
    this->multiplicity = _multiplicity;
}

bool PackingProblem::load(QString fileName, QString fileType, qreal scale, qreal auxScale) {
    if(fileType != "esicup" && fileType != "cfrefp") {
        QFileInfo info(fileName);
        if(info.suffix() == "xml") fileType = "esicup";
        else if(info.suffix() == "txt") fileType = "cfrefp";
    }
    if(fileType == "esicup")
        return loadEsicup(fileName);
    if(fileType == "cfrefp")
        return loadCFREFP(fileName, scale, auxScale);
    return false;
}

bool checkIfSingleContour(POLYBOOLEAN::PAREA *area, QString name) {
    if(area->f != area) {
        qWarning() << "Warning:" << name << "contains multiple contours! Ignoring extra areas.";
        return false;
    }
//    if(area->NFP_lines != NULL) {
//        qWarning() << "Warning:" << name << "contains degenerated line(s)! Ignoring extra areas.";
//        return false;
//    }
//    if(area->NFP_nodes != NULL) {
//        qWarning() << "Warning:" << name << "contains degenerated node(s)! Ignoring extra areas.";
//        return false;
//    }
    POLYBOOLEAN::PLINE2 *pline = area->cntr;
    if(pline != NULL && pline->next != NULL) {
        qWarning() << "Warning:" << name << "contains hole(s)! Ignoring extra areas.";
        return false;
    }
    return true;
}

POLYBOOLEAN::PAREA *getConcaveSingleContour(POLYBOOLEAN::PAREA *area) {
    if(area->f != area) {
        POLYBOOLEAN::PAREA * R = NULL;
        POLYBOOLEAN::PLINE2 *polyNode;
        POLYBOOLEAN::PAREA *areaNode = area;
        do {
            POLYBOOLEAN::PAREA * B = NULL;
            polyNode = areaNode->cntr;
            POLYBOOLEAN::PAREA::InclPline(&B, polyNode);
            POLYBOOLEAN::PAREA::Boolean(R, B, &R, POLYBOOLEAN::PAREA::UN);
            delete(B);
            areaNode = areaNode->f;
        } while(areaNode!=area);
        R->NFP_lines = NULL; R->NFP_nodes = NULL; // FIXME: Delete
        return R;
    }
    return area;
}

void Polygon::fromPolybool(POLYBOOLEAN::PAREA *area, qreal scale) {
    checkIfSingleContour(area, name);

	if(area->cntr != NULL) {
		POLYBOOLEAN::VNODE2 *vn = area->cntr->head;
		do {
			this->push_back(QPointF((qreal)vn->g.x/scale, (qreal)vn->g.y/scale));
		} while( (vn = vn->next) != area->cntr->head);
	}

    // FIXME: Test
    for(POLYBOOLEAN::PLINE2 *NFPpline = area->NFP_lines; NFPpline != NULL; NFPpline = NFPpline->next) {
        POLYBOOLEAN::VNODE2 *vn1, *vn2;
        vn1 = NFPpline->head; vn2 = vn1->next;
        QPair<QPointF, QPointF> curLine;
        curLine.first  = QPoint((qreal)vn1->g.x/scale, (qreal)vn1->g.y/scale);
        curLine.second = QPoint((qreal)vn2->g.x/scale, (qreal)vn2->g.y/scale);
        this->degLines.push_back(curLine);
    }
    for(POLYBOOLEAN::PLINE2 *NFPnode = area->NFP_nodes; NFPnode != NULL; NFPnode = NFPnode->next) {
        POLYBOOLEAN::VNODE2 *vn1 = NFPnode->head;
        this->degNodes.push_back(QPoint((qreal)vn1->g.x/scale, (qreal)vn1->g.y/scale));
    }
}

bool PackingProblem::loadCFREFP(QString &fileName, qreal scale, qreal auxScale) {
    std::vector<std::shared_ptr<cShape> > shapes;
    std::shared_ptr<cRectangularContainer> container;

    FILE *f = fopen(fileName.toUtf8().constData(), "rt");
    if(f==NULL) {
        qCritical() << "Puzzle file not found";
        return false;
    }
    container = readProblemInstance(f, shapes, scale);
    fclose(f);

    unsigned int polygonid = 0;
    QString containerName;
    QMap<int,QString> pieceNames;

    // Add container
    {
        std::shared_ptr<Container> curContainer(new Container("board0",1));
        containerName = "polygon" + QString::number(polygonid); polygonid++;        
        std::shared_ptr<Polygon> curPolygon(new Polygon(containerName));
        curPolygon->fromPolybool(container->getShape()->getInnerData(), scale*auxScale);
        curContainer->setPolygon(curPolygon);
        this->containers.push_back(curContainer);
    }

    // Add pieces
    for(std::vector<std::shared_ptr<cShape>>::const_iterator it = shapes.cbegin(); it != shapes.cend(); it++) {
        std::shared_ptr<cShape> curShape = *it;
        std::shared_ptr<Piece> curPiece(new Piece("piece" + QString::number(curShape->getId()), curShape->getMultiplicity()));
        for(unsigned int i = 0; i < curShape->getAnglesCount(); i++){
            curPiece->addOrientation(curShape->getAngle(i));
        }
        curShape->getPlacement()->updatePlacement(cVector(0,0));
        curShape->setRot(0); curShape->setdRot(0);
        QString pieceName = "polygon" + QString::number(polygonid); polygonid++;
        pieceNames.insert(curShape->getId(), pieceName);
        std::shared_ptr<Polygon> curPolygon(new Polygon(pieceName));
        curPolygon->fromPolybool(getConcaveSingleContour(curShape->getTranslatedPolygon()->getInnerData()), scale*auxScale);
        curPiece->setPolygon(curPolygon);
        this->pieces.push_back(curPiece);
    }

    // Add nofit polygons
    polygonid = 0;
    for(std::vector<std::shared_ptr<cShape>>::const_iterator sit = shapes.cbegin(); sit != shapes.cend(); sit++) {
        int staticAnglesCount = (*sit)->getAnglesCount();
        int staticId = (*sit)->getId();
        bool noStaticAngles = false;
        if(staticAnglesCount == 0) {staticAnglesCount = 1; noStaticAngles = true;}
        for(int i = 0; i < staticAnglesCount; i++){
            int sangle = noStaticAngles ? 0 : (*sit)->getAngle(i);
            for(std::vector<std::shared_ptr<cShape>>::const_iterator oit = shapes.cbegin(); oit != shapes.cend(); oit++) {
                int orbitingAnglesCount = (*oit)->getAnglesCount();
                int orbitingId = (*oit)->getId();
                bool noOrbitingAngles = false;
                if(orbitingAnglesCount == 0) {orbitingAnglesCount = 1; noOrbitingAngles = true;}
                for(int j = 0; j < orbitingAnglesCount; j++){
                    int oangle = noOrbitingAngles ? 0 : (*oit)->getAngle(j);
                    // -> Create Descriptor
                    std::shared_ptr<NoFitPolygon> curNFP(new NoFitPolygon);
                    curNFP->setStaticName(pieceNames[staticId]);
                    curNFP->setOrbitingName(pieceNames[orbitingId]);
                    curNFP->setStaticAngle(sangle);
                    curNFP->setOrbitingAngle(oangle);
                    QString polygonName = "nfpPolygon" + QString::number(polygonid); polygonid++;
                    curNFP->setName(polygonName);

                    // --> Get Nofit Polygon
                    std::shared_ptr<cShape> staticShape = *sit;
                    std::shared_ptr<cShape> orbitingShape = *oit;
                    staticShape->getPlacement()->updatePlacement(cVector(0,0));
                    staticShape->setRot(sangle); staticShape->setdRot(sangle);
                    std::shared_ptr<cPolygon> staticPolygon = staticShape->getTranslatedPolygon();
                    orbitingShape->getPlacement()->updatePlacement(cVector(0,0));
                    orbitingShape->setRot(oangle); orbitingShape->setdRot(oangle);
                    std::shared_ptr<cPolygon> orbitingPolygon = orbitingShape->getTranslatedPolygon();
                    std::shared_ptr<cPolygon> nfp = orbitingPolygon->getNoFitPolygon(staticPolygon);

                    std::shared_ptr<Polygon> curPolygon(new Polygon(polygonName));
                    curPolygon->fromPolybool(nfp->getInnerData(), scale*auxScale);
                    curNFP->setPolygon(curPolygon);
                    this->nofitPolygons.push_back(curNFP);
                }

            }
        }
    }

    // Add innerfit polygons
    polygonid = 0;
    for(std::vector<std::shared_ptr<cShape>>::const_iterator oit = shapes.cbegin(); oit != shapes.cend(); oit++) {
        int orbitingAnglesCount = (*oit)->getAnglesCount();
        int orbitingId = (*oit)->getId();
        bool noOrbitingAngles = false;
        if(orbitingAnglesCount == 0) {orbitingAnglesCount = 1; noOrbitingAngles = true;}
        for(int j = 0; j < orbitingAnglesCount; j++){
            int oangle = noOrbitingAngles ? 0 : (*oit)->getAngle(j);
            // -> Create Descriptor
            std::shared_ptr<InnerFitPolygon> curIFP(new InnerFitPolygon);
            curIFP->setStaticName(containerName);
            curIFP->setOrbitingName(pieceNames[orbitingId]);
            curIFP->setStaticAngle(0);
            curIFP->setOrbitingAngle(oangle);
            QString polygonName = "ifpPolygon" + QString::number(polygonid); polygonid++;
            curIFP->setName(polygonName);

            // --> Get Inner Polygon
            std::shared_ptr<cShape> orbitingShape = *oit;
            orbitingShape->getPlacement()->updatePlacement(cVector(0,0));
            orbitingShape->setRot(oangle); orbitingShape->setdRot(oangle);
            std::shared_ptr<cPolygon> orbitingPolygon = orbitingShape->getTranslatedPolygon();
            std::shared_ptr<cPolygon> ifp = container->getInnerFitPolygon(orbitingPolygon);

            std::shared_ptr<Polygon> curPolygon(new Polygon(polygonName));
            curPolygon->fromPolybool(ifp->getInnerData(), scale*auxScale);
            curIFP->setPolygon(curPolygon);
            this->innerfitPolygons.push_back(curIFP);
        }
    }

    return true;
}

bool PackingProblem::copyHeader(QString fileName) {
    QFile file(fileName);
    if (!file.open(QFile::ReadOnly | QFile::Text)) {
        qCritical() << "Error: Cannot read file"
                    << ": " << qPrintable(file.errorString());
        return false;
    }

    QXmlStreamReader xml;
    xml.setDevice(&file);
    while (!xml.atEnd()) {
        xml.readNext();

        if(xml.name()=="name" && xml.tokenType() == QXmlStreamReader::StartElement) this->name = xml.readElementText();
        if(xml.name()=="author" && xml.tokenType() == QXmlStreamReader::StartElement) this->author = xml.readElementText();
        if(xml.name()=="date" && xml.tokenType() == QXmlStreamReader::StartElement) this->date = xml.readElementText();
        if(xml.name()=="description" && xml.tokenType() == QXmlStreamReader::StartElement) this->description = xml.readElementText();
    }

    if (xml.hasError()) {
        // do error handling
    }

    file.close();
    return true;
}

bool PackingProblem::loadEsicup(QString &fileName) {
    QFile file(fileName);
    if (!file.open(QFile::ReadOnly | QFile::Text)) {
        qCritical() << "Error: Cannot read file"
                    << ": " << qPrintable(file.errorString());
        return false;
    }

    QXmlStreamReader xml;

    std::shared_ptr<Piece> curPiece;
    std::shared_ptr<Container> curContainer;
    QMap<QString, std::shared_ptr<PackingComponent>> pieceNameMap;
    QMap<QString, std::shared_ptr<Polygon> > polygonsTempSet;
    std::shared_ptr<Polygon> curPolygon;
    std::shared_ptr<GeometricTool> curGeometricTool;
    //    PLINE2 *curContour = NULL;
    ReadStates curState = NEUTRAL;

    xml.setDevice(&file);
    while (!xml.atEnd()) {
        xml.readNext();

        if(xml.name()=="name" && xml.tokenType() == QXmlStreamReader::StartElement) this->name = xml.readElementText();
        if(xml.name()=="author" && xml.tokenType() == QXmlStreamReader::StartElement) this->author = xml.readElementText();
        if(xml.name()=="date" && xml.tokenType() == QXmlStreamReader::StartElement) this->date = xml.readElementText();
        if(xml.name()=="description" && xml.tokenType() == QXmlStreamReader::StartElement) this->description = xml.readElementText();

        // Determine state
        if(xml.name()=="boards" && xml.tokenType() == QXmlStreamReader::StartElement) curState = BOARDS_READING;
        if(xml.name()=="lot" && xml.tokenType() == QXmlStreamReader::StartElement) curState = LOT_READING;
        if( (xml.name()=="boards" || xml.name()=="lot") && xml.tokenType() == QXmlStreamReader::EndElement) curState = NEUTRAL;

        // Read new piece
        if(xml.name()=="piece" && xml.tokenType() == QXmlStreamReader::StartElement) {
            if(curState == BOARDS_READING) {
                curContainer = std::shared_ptr<Container>(new Container(xml.attributes().value("id").toString(), xml.attributes().value("quantity").toInt()));
                this->containers.push_back(curContainer);
            }
            if(curState == LOT_READING) {
                curPiece = std::shared_ptr<Piece>(new Piece(xml.attributes().value("id").toString(), xml.attributes().value("quantity").toInt()));
                this->pieces.push_back(curPiece);
            }
        }

        // Add piece general information
        if(xml.name()=="component" && xml.tokenType() == QXmlStreamReader::StartElement) {
            if(curState == BOARDS_READING)
                pieceNameMap.insert(xml.attributes().value("idPolygon").toString(), curContainer);
            if(curState == LOT_READING)
                pieceNameMap.insert(xml.attributes().value("idPolygon").toString(), curPiece);
        }

        // Add piece orientation
        if(xml.name()=="enumeration" && xml.tokenType() == QXmlStreamReader::StartElement) {
            curPiece->addOrientation(xml.attributes().value("angle").toInt());
        }

        // Read new polygon
        if(xml.name()=="polygon" && xml.tokenType() == QXmlStreamReader::StartElement) {
            // TODO
            curPolygon = std::shared_ptr<Polygon>(new Polygon(xml.attributes().value("id").toString()));
            // Associate with corresponding piece
            QString polygonName = curPolygon->getName(); //xml.attributes().value("id").toString();
            QMap<QString, std::shared_ptr<PackingComponent>>::iterator it;
            if( (it = pieceNameMap.find(polygonName)) != pieceNameMap.end()) {
                (*it)->setPolygon(curPolygon);
            }
            else polygonsTempSet.insert(curPolygon->getName(), curPolygon);
        }

        // Add vertex to contour
        if(xml.name()=="segment" && xml.tokenType() == QXmlStreamReader::StartElement) {
            QPointF vertex;
            vertex.setX(xml.attributes().value("x0").toDouble());
            vertex.setY(xml.attributes().value("y0").toDouble());
            curPolygon->push_back(vertex);
        }

        // Process nofit polygon informations
        if(xml.name()=="nfp" && xml.tokenType() == QXmlStreamReader::StartElement) {
            curGeometricTool = std::shared_ptr<NoFitPolygon>(new NoFitPolygon);
        }
        if(xml.name()=="ifp" && xml.tokenType() == QXmlStreamReader::StartElement) {
            curGeometricTool = std::shared_ptr<InnerFitPolygon>(new InnerFitPolygon);
        }
        if(xml.name()=="staticPolygon" && xml.tokenType() == QXmlStreamReader::StartElement) {
            curGeometricTool->setStaticAngle(xml.attributes().value("angle").toInt());
            curGeometricTool->setStaticName(xml.attributes().value("idPolygon").toString());
        }
        if(xml.name()=="orbitingPolygon" && xml.tokenType() == QXmlStreamReader::StartElement) {
            curGeometricTool->setOrbitingAngle(xml.attributes().value("angle").toInt());
            curGeometricTool->setOrbitingName(xml.attributes().value("idPolygon").toString());
        }
        if(xml.name()=="resultingPolygon" && xml.tokenType() == QXmlStreamReader::StartElement) {
            curGeometricTool->setName(xml.attributes().value("idPolygon").toString());
        }
        if(xml.name()=="nfp" && xml.tokenType() == QXmlStreamReader::EndElement)
            this->nofitPolygons.push_back(std::static_pointer_cast<NoFitPolygon>(curGeometricTool));
        if(xml.name()=="ifp" && xml.tokenType() == QXmlStreamReader::EndElement)
            this->innerfitPolygons.push_back(std::static_pointer_cast<InnerFitPolygon>(curGeometricTool));
    }

    if (xml.hasError()) {
        // do error handling
    }

    for(QList<std::shared_ptr<NoFitPolygon>>::const_iterator it = this->cnfpbegin(); it != this->cnfpend(); it++) {
        QMap<QString, std::shared_ptr<Polygon> >::iterator resultingPolygonIt = polygonsTempSet.find((*it)->getName());
        if( resultingPolygonIt != polygonsTempSet.end())
            (*it)->setPolygon(*resultingPolygonIt);
        else return false; // do error handling
    }

    for(QList<std::shared_ptr<InnerFitPolygon>>::const_iterator it = this->cifpbegin(); it != this->cifpend(); it++) {
        QMap<QString, std::shared_ptr<Polygon> >::iterator resultingPolygonIt = polygonsTempSet.find((*it)->getName());
        if( resultingPolygonIt != polygonsTempSet.end())
            (*it)->setPolygon(*resultingPolygonIt);
        else return false; // do error handling
    }

    file.close();
    return true;
}

void processXMLCommands(QStringList &commands, QXmlStreamWriter &stream) {
    for(QStringList::Iterator it = commands.begin(); it != commands.end(); it++) {
        QString curCommand = *it;
        if(curCommand == "writeStartElement")
            stream.writeStartElement(*(++it));
        else if(curCommand == "writeAttribute") {
            QString attName = *(++it);
            stream.writeAttribute(attName, *(++it));
        }
        else if(curCommand == "writeTextElement") {
            QString attName = *(++it);
            stream.writeTextElement(attName, *(++it));
        }
        else if(curCommand == "writeEndElement")
            stream.writeEndElement();
    }
}

bool PackingProblem::save(QString fileName) {
    QFile file(fileName);
    if(!file.open(QIODevice::WriteOnly)) {
        qCritical() << "Error: Cannot create output file" << fileName << ": " << qPrintable(file.errorString());
        return false;
    }

    QXmlStreamWriter stream;
    stream.setDevice(&file);
    stream.setAutoFormatting(true);
    stream.writeStartDocument();

    stream.writeStartElement("nesting");
    stream.writeAttribute("xmlns", "http://globalnest.fe.up.pt/nesting");
    stream.writeTextElement("name", this->name);
    stream.writeTextElement("author", this->author);
    stream.writeTextElement("date", this->date);
    stream.writeTextElement("description", this->description);
    stream.writeTextElement("verticesOrientation", "clockwise");
    stream.writeTextElement("coordinatesOrigin", "up-left");

    // --> Write problem
    stream.writeStartElement("problem");
    stream.writeStartElement("boards");
    for(QList<std::shared_ptr<Container>>::const_iterator it = this->ccbegin(); it != this->ccend(); it++) {
        QStringList containerCommand = (*it)->getXML();
        processXMLCommands(containerCommand, stream);
    }
    stream.writeEndElement(); // boards
    stream.writeStartElement("lot");
    for(QList<std::shared_ptr<Piece>>::const_iterator it = this->cpbegin(); it != this->cpend(); it++) {
        QStringList pieceCommand = (*it)->getXML();
        processXMLCommands(pieceCommand, stream);
    }
    stream.writeEndElement(); // lot
    stream.writeEndElement(); // problem

    // --> Write polygons
    // FIXME: Duplicated polygons?
    stream.writeStartElement("polygons");
    for(QList<std::shared_ptr<Container>>::const_iterator it = this->ccbegin(); it != this->ccend(); it++) {
        QStringList containerPolCommand = (*it)->getPolygon()->getXML();
        processXMLCommands(containerPolCommand, stream);
    }
    for(QList<std::shared_ptr<Piece>>::const_iterator it = this->cpbegin(); it != this->cpend(); it++) {
        QStringList piecePolCommand = (*it)->getPolygon()->getXML();
        processXMLCommands(piecePolCommand, stream);
    }
    for(QList<std::shared_ptr<NoFitPolygon>>::const_iterator it = this->cnfpbegin(); it != this->cnfpend(); it++) {
        QStringList nfpPolCommand = (*it)->getPolygon()->getXML();
        processXMLCommands(nfpPolCommand, stream);
    }
    for(QList<std::shared_ptr<InnerFitPolygon>>::const_iterator it = this->cifpbegin(); it != this->cifpend(); it++) {
        QStringList ifpPolCommand = (*it)->getPolygon()->getXML();
        processXMLCommands(ifpPolCommand, stream);
    }
    stream.writeEndElement(); // polygons

    // --> Write nfps and ifps
    stream.writeStartElement("nfps");
    for(QList<std::shared_ptr<NoFitPolygon>>::const_iterator it = this->cnfpbegin(); it != this->cnfpend(); it++) {
        QStringList nfpCommand = (*it)->getXML();
        processXMLCommands(nfpCommand, stream);
    }
    stream.writeEndElement(); // nfps
    stream.writeStartElement("ifps");
    for(QList<std::shared_ptr<InnerFitPolygon>>::const_iterator it = this->cifpbegin(); it != this->cifpend(); it++) {
        QStringList ifpCommand = (*it)->getXML();
        processXMLCommands(ifpCommand, stream);
    }
    stream.writeEndElement(); // ifps

    // --> Write raster results
    if(!this->rasterNofitPolygons.empty()) {
        stream.writeStartElement("raster");
        for(QList<std::shared_ptr<RasterNoFitPolygon>>::const_iterator it = this->crnfpbegin(); it != this->crnfpend(); it++) {
            QStringList rnfpCommand = (*it)->getXML();
            processXMLCommands(rnfpCommand, stream);
        }
        for(QList<std::shared_ptr<RasterInnerFitPolygon>>::const_iterator it = this->crifpbegin(); it != this->crifpend(); it++) {
            QStringList rifpCommand = (*it)->getXML();
            processXMLCommands(rifpCommand, stream);
        }
        stream.writeEndElement(); // raster
    }

    stream.writeEndElement(); // nesting
    stream.writeEndDocument();
    file.close();

    return true;
}

