#include "weightviewer.h"
#include <QImage>
#include <QGraphicsPixmapItem>
#include <QMouseEvent>
#include <QTimer>
#include <QDebug>

using namespace RASTERVORONOIPACKING;

WeightViewer::WeightViewer(QWidget *parent) :
    QGraphicsView(parent),
    pairSelected(false)
{
    QGraphicsScene *scene = new QGraphicsScene;
    curMap = new QGraphicsPixmapItem;
    scene->addItem(curMap);
    setScene(scene);
    currentScale = 1.0;
    scale(currentScale,currentScale);
}

 void WeightViewer::init(int _numItems) {
     numItems = _numItems;
     for(int i = 0; i < numItems; i++) {
         for(int j = 0; j < numItems; j++) {
             GraphicsIndexedRectItem *curCell = new GraphicsIndexedRectItem(QRectF(i,j,1,1), i, j);
             curCell->setPen(Qt::NoPen);
             //curCell->setBrush(QColor(100,100,255,100));
             curCell->setBrush(Qt::NoBrush);
             curCell->setParentItem(curMap);
             cells.insert(QPair<int,int>(i,j), curCell);
         }
     }
 }

void WeightViewer::updateImage() {
    pmap = QPixmap::fromImage(weights->getImage(numItems));
    curMap->setPixmap(pmap);
    for(int i = 0; i < numItems; i++)
        for(int j = 0; j < numItems; j++)
            if(i != j)
                cells[QPair<int,int>(i,j)]->setToolTip(QString::number(weights->getWeight(i,j),'g',2) + " (" + QString::number(i) + ", " + QString::number(j) + ")");
}

void WeightViewer::resizeEvent(QResizeEvent *) {
    scale(1/currentScale, 1/currentScale);
    qreal horizontalScale = (qreal)this->width()/(qreal)pmap.width();
    qreal verticalScale = (qreal)this->height()/(qreal)pmap.height();
    currentScale = horizontalScale < verticalScale ? horizontalScale : verticalScale;
    scale(currentScale, currentScale);
//    if(!pmap.isNull()) curMap->setPixmap(pmap.scaled(size(),Qt::KeepAspectRatio));
}

void WeightViewer::mousePressEvent(QMouseEvent * event) {
    GraphicsIndexedRectItem *pickedCell = (GraphicsIndexedRectItem*)itemAt(event->pos());

    if(pickedCell != 0 && !pairSelected && pickedCell->getIndex().x() != pickedCell->getIndex().y()) {
        pairSelected = true;
        emit selectionChanged(pickedCell->getIndex().x(), pickedCell->getIndex().y());

        GraphicsIndexedRectItem *pickedCell2 = cells[QPair<int,int>(pickedCell->getIndex().y(),pickedCell->getIndex().x())];

        pickedCell->setBrush(QColor(255,0,255));
        pickedCell2->setBrush(QColor(255,0,255));
        QTimer *timer = new QTimer(this);
        timer->setSingleShot(true);
        connect(timer, &QTimer::timeout, [pickedCell, pickedCell2, this](){
            pickedCell->setBrush(Qt::NoBrush); pickedCell2->setBrush(Qt::NoBrush);
            pairSelected = false;});
        timer->start(2000);
    }
    QGraphicsView::mousePressEvent(event);
}
