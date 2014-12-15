#ifndef WEIGHTVIEWER_H
#define WEIGHTVIEWER_H

#include <QGraphicsView>
#include <QPixmap>
#include "graphicsindexedrectitem.h"
#include "../raster/glsweightset.h"

class QGraphicsPixmapItem;

namespace Ui {
class WeightViewer;
}

class WeightViewer : public QGraphicsView
{
    Q_OBJECT
public:
    explicit WeightViewer(QWidget *parent = 0);
    void init(int _numItems);
    void setWeights(std::shared_ptr<RASTERVORONOIPACKING::GlsWeightSet> w) {weights = w;}

signals:
    void selectionChanged(int i, int j);

public slots:
    void updateImage();

protected:
    void mousePressEvent(QMouseEvent * event);
    void resizeEvent(QResizeEvent *);

private:
    QGraphicsPixmapItem *curMap;
    QPixmap pmap;
    std::shared_ptr<RASTERVORONOIPACKING::GlsWeightSet> weights;
    int numItems;
    QHash<QPair<int,int>, GraphicsIndexedRectItem*> cells;
    int pairSelected;
    qreal currentScale;
};

#endif // WEIGHTVIEWER_H
