#ifndef PACKINGITEM_H
#define PACKINGITEM_H

#include <QGraphicsPolygonItem>
#include <QDebug>

class PackingItem : public QGraphicsPolygonItem
{
public:
    PackingItem ( QGraphicsItem * parent = 0 );
    PackingItem (const QPolygonF & polygon, int _id, QGraphicsItem * parent = 0 );
    ~PackingItem() {};

    void setContainer(QRectF &_container, int angle = 0);
    QRectF getInnerFitRectangle() {return innerFit;}

    void clearAngles() {angles.clear();}
    void addAngle(int angle) {angles.push_back(angle);}
    int getAngle(int i) {return angles[i];}
    int getAnglesCount() {return angles.size();}
    QVector<int>::const_iterator anglesBegin() {return angles.cbegin();}
    QVector<int>::const_iterator anglesEnd() {return angles.cend();}
    QVector<int> getAnglesList() {return angles;}
    void setGridSize(qreal grid) {gridSize = grid;}
    qreal getGridSize() {return gridSize;}
    int getId() {return id;}

    void changeOrientation(int angleId);
    void rotateNext();

    int getCurAngle() {return curAngle;}

protected:
    QVariant itemChange ( GraphicsItemChange change, const QVariant & value );
//    void mouseMoveEvent ( QGraphicsSceneMouseEvent * event );
    void mousePressEvent ( QGraphicsSceneMouseEvent * event );

private:
    QRectF innerFit;
    QRectF container;
    QVector<int> angles;
    qreal gridSize;
    int id;
    int curAngle;
};

#endif // PACKINGITEM_H
