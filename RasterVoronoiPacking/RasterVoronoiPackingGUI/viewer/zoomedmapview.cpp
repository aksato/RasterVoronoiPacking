#include "zoomedmapview.h"
#include <QImage>
#include <QGraphicsPixmapItem>
#include <QDebug>

ZoomedMapView::ZoomedMapView(QWidget *parent) : QGraphicsView(parent)
{
    QGraphicsScene *scene = new QGraphicsScene;
    curMap = new QGraphicsPixmapItem;
    horizontalBar = new QGraphicsRectItem; horizontalBar->setVisible(false);  horizontalBar->setBrush(QColor(240,240,240)); horizontalBar->setPen(QPen(QColor(240,240,240), 0));
    verticalBar = new QGraphicsRectItem; verticalBar->setVisible(false); verticalBar->setBrush(QColor(240,240,240)); verticalBar->setPen(QPen(QColor(240,240,240), 0));
    scene->addItem(curMap);
    scene->addItem(horizontalBar);
    scene->addItem(verticalBar);
    setScene(scene);
    currentScale = 1.0;
    scale(currentScale,-currentScale);
}

void ZoomedMapView::resizeEvent(QResizeEvent *) {
    scale(1/currentScale, -1/currentScale);
    qreal horizontalScale = (qreal)this->width()/(qreal)pmap.width();
    qreal verticalScale = (qreal)this->height()/(qreal)pmap.height();
    currentScale = horizontalScale < verticalScale ? horizontalScale : verticalScale;
    scale(currentScale, -currentScale);
}

void ZoomedMapView::mousePressEvent(QMouseEvent * event) {
    QGraphicsView::mousePressEvent(event);
}

void ZoomedMapView::updateMap(std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> map, QPoint &centerPoint) {
	int deltaX = -map->getReferencePoint().x() + size / 2 - centerPoint.x();
	int deltaY = -map->getReferencePoint().y() + size / 2 - centerPoint.y();
        QPoint deltaPt(deltaX, deltaY);
        QPixmap pixMap = QPixmap::fromImage(map->getZoomImage(size, size, deltaPt));
	setImage(pixMap);
	QPoint refPoint = map->getReferencePoint();
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			cells[QPair<int, int>(i, j)]->setToolTip("(" + QString::number(ratio *(-refPoint.x() + i - deltaX)) + ", " + QString::number(ratio *(-refPoint.y() + j - deltaY)) + ")");
}

void ZoomedMapView::init(int mapSize, qreal _ratio) {
	size = mapSize;
	this->ratio = _ratio;
	for (int i = 0; i < mapSize; i++) {
		for (int j = 0; j < mapSize; j++) {
			GraphicsIndexedRectItem *curCell = new GraphicsIndexedRectItem(QRectF(i, j, 1, 1), i, j);
			curCell->setPen(QPen(Qt::black, 0));
			curCell->setBrush(Qt::NoBrush);
			curCell->setParentItem(curMap);
			cells.insert(QPair<int, int>(i, j), curCell);
		}
	}
}

void ZoomedMapView::setImage(QPixmap _pmap){
    pmap = _pmap.copy();
    curMap->setPixmap(pmap);
}

void ZoomedMapView::setValidArea(QRect validArea) {
    verticalBar->setVisible(false); horizontalBar->setVisible(false);
    if(validArea.width() != pmap.width()) {
        verticalBar->setRect(QRectF(validArea.width(), 0, pmap.width() - validArea.width(), pmap.height()));
        verticalBar->setVisible(true);
    }
    if(validArea.height() != pmap.height()) {
        horizontalBar->setRect(QRectF(0, validArea.height(), pmap.height(), pmap.height() - validArea.height()));
        horizontalBar->setVisible(true);
    }
}
