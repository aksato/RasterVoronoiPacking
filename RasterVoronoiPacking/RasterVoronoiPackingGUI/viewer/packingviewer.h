#ifndef PACKINGVIEWER_H
#define PACKINGVIEWER_H

#include <QWidget>
#include <QGraphicsView>
#include <QHash>
#include <QVector>
#include <QPolygonF>
#include <QStatusBar>
//#include "raster/rasternofitpolygon.h"
#include "raster/rasterpackingproblem.h"
#include "raster/rasterpackingsolution.h"
#include "raster/totaloverlapmap.h"
#include "packingitem.h"
namespace RASTERPACKING {class PackingProblem;}
class QGraphicsPolygonItem;
namespace Ui {
class PackingViewer;
}
class QGraphicsScene;
class QGraphicsEllipseItem;

class PackingViewer : public QGraphicsView
{
    Q_OBJECT

public:
    explicit PackingViewer(QWidget *parent = 0);
    ~PackingViewer();

    void createGraphicItems(RASTERPACKING::PackingProblem &problem);
    void getCurrentSolution(RASTERVORONOIPACKING::RasterPackingSolution &solution, qreal scale);
    void getCurrentSolution(RASTERVORONOIPACKING::RasterPackingSolution &solution);
	void showTotalOverlapMap(std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> newMap, qreal scale = 1);

    void setStatusBar(QStatusBar *bar) {statusBar = bar;}
    int getTotalItems() {return pieces.size();}
    QVector<int> getItemAngles(int id) {return pieces[id]->getAnglesList();}
    int getItemAngle(int id) {return pieces[id]->getCurAngle();}
    QPointF getItemPos(int id) {return pieces[id]->pos();}
    void setItemPos(QPointF pos, int id) {pieces[id]->setPos(pos);}// TOREMOVE
    qreal getScale() {return rasterScale;}
    int getCurrentItemId() {return currentPieceId;}
    void setRasterScale(qreal newScale) {
        this->rasterScale = newScale;
        changeGridSize(newScale);
    }

    void changeGridSize(qreal newScale) {
        for(int i = 0; i < pieces.size(); i++)
            pieces[i]->setGridSize(1/newScale);
    }

    void disableItemSelection();
    void enableItemSelection();

public slots:
    void setCurrentSolution(const RASTERVORONOIPACKING::RasterPackingSolution &solution);
    void setCurrentSolution(const RASTERVORONOIPACKING::RasterPackingSolution &solution, qreal scale);
    void setZoomPosition(int pos);
    void setSelectedItem(int id);
    void setCurrentOrientation(int angle);
    void setCurrentXCoord(double xpos);
    void setCurrentYCoord(double ypos);
    void highlightPair(int id1, int id2);
	void recreateContainerGraphics(int pixelWidth);
	void recreateContainerGraphics(int pixelWidth, int pixelHeight);

signals:
    void zoomChanged(int pos);
    void selectedItemChanged(int id);
    void currentOrientationChanged(int angle);
    void currentPositionChanged(QPointF pos);

protected:
#ifndef QT_NO_WHEELEVENT
    void wheelEvent(QWheelEvent *event);
#endif
    void keyPressEvent(QKeyEvent *event);
    void mousePressEvent(QMouseEvent * event);
    void mouseMoveEvent ( QMouseEvent * event );


private:
    void highlightItem(int id);

    Ui::PackingViewer *ui;

    QGraphicsScene* mainScene;
    QVector<PackingItem*> pieces;
    QGraphicsPolygonItem* container;
    QGraphicsEllipseItem *originItem;
    QGraphicsPixmapItem *curMap;
    int currentPieceId;

    qreal rasterScale;
    int zoomPosition;
    bool itemMovementLocked;

    QStatusBar *statusBar;
};

#endif // PACKINGVIEWER_H
