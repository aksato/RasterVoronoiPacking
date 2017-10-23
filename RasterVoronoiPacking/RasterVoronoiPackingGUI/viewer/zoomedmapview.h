#ifndef ZOOMEDMAPVIEW_H
#define ZOOMEDMAPVIEW_H

#include <QGraphicsView>
#include <QPixmap>
#include "graphicsindexedrectitem.h"
#include "../common/raster/totaloverlapmap.h"
#include <memory>

class QGraphicsPixmapItem;

class ZoomedMapView : public QGraphicsView
{
    Q_OBJECT
public:
    explicit ZoomedMapView(QWidget *parent = 0);
	void init(int mapSize, qreal _ratio);
	void updateMap(std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> map, QPoint &centerPoint);
    void setValidArea(QRect validArea);

protected:
    void mousePressEvent(QMouseEvent * event);
    void resizeEvent(QResizeEvent *);

private:
	void setImage(QPixmap _pmap);

    QGraphicsPixmapItem *curMap;
	QHash<QPair<int, int>, GraphicsIndexedRectItem*> cells;
    QGraphicsRectItem *verticalBar, *horizontalBar;
    QPixmap pmap;
    qreal currentScale;
	qreal ratio;
	int size;
};

#endif // ZOOMEDMAPVIEW_H
