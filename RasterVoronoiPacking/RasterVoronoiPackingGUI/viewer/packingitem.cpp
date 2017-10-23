#include "packingitem.h"
#include <QGraphicsScene>
#include <QDebug>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsItem>
PackingItem::PackingItem(QGraphicsItem * parent) : QGraphicsPolygonItem(parent) , gridSize(1.0) {this->setFlag(ItemSendsGeometryChanges);}
PackingItem::PackingItem (const QPolygonF & polygon, int _id, QGraphicsItem * parent) : QGraphicsPolygonItem(polygon, parent) , gridSize(1.0), id(_id), curAngle(0) {
    this->setFlag(ItemSendsGeometryChanges);
}

void PackingItem::setContainer(QRectF &_container, int angle) {
    this->container = _container;

    QRectF boundingBox = polygon().boundingRect();
    QTransform trans; trans=trans.rotate(angle); boundingBox=trans.mapRect(boundingBox);
    innerFit.setX(0); innerFit.setY(0);
    innerFit.setWidth(container.width()-boundingBox.width());
    innerFit.setHeight(container.height()-boundingBox.height());
    innerFit.translate(-boundingBox.topLeft());
}

QVariant PackingItem::itemChange(GraphicsItemChange change, const QVariant &value)
{
    if (change == ItemPositionChange && scene()) {
        // value is the new position.
        QPointF newPos = value.toPointF();
        QRectF rect = innerFit;

        QPointF deltaPos = newPos - pos();
//        if(deltaPos.x() != 0) newPos.setX(floor(newPos.x()/gridSize)*gridSize);
//        if(deltaPos.y() != 0) newPos.setY(floor(newPos.y()/gridSize)*gridSize);
        if(deltaPos.x() != 0) newPos.setX(qRound(newPos.x()/gridSize)*gridSize);
        if(deltaPos.y() != 0) newPos.setY(qRound(newPos.y()/gridSize)*gridSize);

        //if (!rect.contains(newPos)) {
        //    // Keep the item inside the scene rect.
        //    newPos.setX(qMin(rect.right(), qMax(newPos.x(), rect.left())));
        //    newPos.setY(qMin(rect.bottom(), qMax(newPos.y(), rect.top())));
        //}
        return newPos;
    }
    return QGraphicsItem::itemChange(change, value);
}

void PackingItem::mousePressEvent ( QGraphicsSceneMouseEvent * event ) {
    QGraphicsItem::mousePressEvent(event);
}


void PackingItem::changeOrientation(int angleId) {
    if(this->curAngle != angleId) {
        this->curAngle = angleId;

        setRotation(getAngle(this->curAngle));
        setContainer(this->container, getAngle(this->curAngle));

        QRectF rect = getInnerFitRectangle();
        QPointF newPos = pos();
        newPos.setX(qMin(rect.right(), qMax(newPos.x(), rect.left())));
        newPos.setY(qMin(rect.bottom(), qMax(newPos.y(), rect.top())));
        setPos(newPos);
    }
}

void PackingItem::rotateNext() {
    changeOrientation( (this->curAngle+1)%angles.size() );
}
