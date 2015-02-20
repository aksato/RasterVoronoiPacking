#ifndef GRAPHICSINDEXEDRECTITEM_H
#define GRAPHICSINDEXEDRECTITEM_H

#include <QGraphicsRectItem>
#include <QPoint>

class GraphicsIndexedRectItem : public QGraphicsRectItem
{
public:
    GraphicsIndexedRectItem(QGraphicsItem * parent = 0 );
    GraphicsIndexedRectItem(const QRectF & rect, int i, int j, QGraphicsItem * parent = 0);

    void setIndex(QPoint pt) {index = pt;}
    void setIndex(int i, int j) {index = QPoint(i,j);}
    QPoint getIndex() {return index;}

private:
    QPoint index;
};

#endif // GRAPHICSINDEXEDRECTITEM_H
