#ifndef ZOOMEDMAPVIEW_H
#define ZOOMEDMAPVIEW_H

#include <QGraphicsView>
#include <QPixmap>
class QGraphicsPixmapItem;

class ZoomedMapView : public QGraphicsView
{
    Q_OBJECT
public:
    explicit ZoomedMapView(QWidget *parent = 0);
    void setImage(QPixmap _pmap);
    void setValidArea(QRect validArea);

protected:
    void mousePressEvent(QMouseEvent * event);
    void resizeEvent(QResizeEvent *);

private:
    QGraphicsPixmapItem *curMap;
    QGraphicsRectItem *verticalBar, *horizontalBar;
    QPixmap pmap;
    qreal currentScale;
};

#endif // ZOOMEDMAPVIEW_H
