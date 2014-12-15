#include "graphicsindexedrectitem.h"

GraphicsIndexedRectItem::GraphicsIndexedRectItem(QGraphicsItem *parent) :
    QGraphicsRectItem(parent)
{
}

GraphicsIndexedRectItem::GraphicsIndexedRectItem(const QRectF & rect, int i, int j, QGraphicsItem * parent) :
    QGraphicsRectItem(rect, parent)
{
    setIndex(i,j);
}
