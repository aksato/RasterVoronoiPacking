#include "packingviewer.h"
#include "ui_packingviewer.h"
#include "../packingproblem.h"
#include "../colormap.h"
#include <QGraphicsPixmapItem>
#include <QWheelEvent>
#include <QScrollBar>
#include <QtCore/qmath.h>
#include <algorithm>
#include <QTime>
#include <QTimer>

using namespace RASTERVORONOIPACKING;

qreal logslider(int position) {
  // position will be between 0 and 100
  qreal minp = 0;
  qreal maxp = 99;

  // The result should be between 100 an 10000000
  qreal minv = qLn(0.1);
  qreal maxv = qLn(100);

  // calculate adjustment factor
  qreal scale = (maxv-minv) / (maxp-minp);

  return qExp (minv + scale*((qreal)position-minp));
}

int logposition(qreal value) {
   // set minv, ... like above
   // ...

    // position will be between 0 and 100
    qreal minp = 0;
    qreal maxp = 99;

    // The result should be between 100 an 10000000
    qreal minv = qLn(0.1);
    qreal maxv = qLn(100);

    // calculate adjustment factor
    qreal scale = (maxv-minv) / (maxp-minp);

   return (int)((qLn(value)-minv) / scale + minp);
}

PackingViewer::PackingViewer(QWidget *parent) :
    QGraphicsView(parent),
    ui(new Ui::PackingViewer)
{
    ui->setupUi(this);
    currentPieceId = -1;
    mainScene = new QGraphicsScene;
    this->zoomPosition = logposition(1.0);
    setScene(mainScene);
}

PackingViewer::~PackingViewer()
{
    delete ui;
}

void PackingViewer::setZoomPosition(int pos) {
    if(pos != this->zoomPosition) {
        scale(1/logslider(this->zoomPosition),1/logslider(this->zoomPosition));
        this->zoomPosition = pos;
        scale(logslider(this->zoomPosition),logslider(this->zoomPosition));
        emit zoomChanged(this->zoomPosition);
    }
}

void PackingViewer::wheelEvent(QWheelEvent *event)
{
    if(event->modifiers() & Qt::ControlModifier) {
        if(event->delta() > 0) {
            if(this->zoomPosition != 99) {
                scale(1/logslider(this->zoomPosition),1/logslider(this->zoomPosition));
                this->zoomPosition++;
                scale(logslider(this->zoomPosition),logslider(this->zoomPosition));
                emit zoomChanged(this->zoomPosition);
            }
        }
        else {
            if(this->zoomPosition != 0) {
                scale(1/logslider(this->zoomPosition),1/logslider(this->zoomPosition));
                this->zoomPosition--;
                scale(logslider(this->zoomPosition),logslider(this->zoomPosition));
                emit zoomChanged(this->zoomPosition);
            }
        }
//        scaleView(pow((double)2, event->delta() / 240.0));
    }
    else if(event->modifiers() & Qt::ShiftModifier)
        this->horizontalScrollBar()->setValue(this->horizontalScrollBar()->value() - event->delta()/ 2.0);
    else
        this->verticalScrollBar()->setValue(this->verticalScrollBar()->value() - event->delta()/ 2.0);
}

void PackingViewer::keyPressEvent(QKeyEvent *event) {
    if(event->modifiers() & Qt::ControlModifier) {
        if(event->key() == Qt::Key::Key_0) {
            scale(1/logslider(this->zoomPosition),1/logslider(this->zoomPosition));
            this->zoomPosition = logposition(rasterScale);
            scale(logslider(this->zoomPosition),logslider(this->zoomPosition));
            emit zoomChanged(this->zoomPosition);
        }
        if(event->key() == Qt::Key::Key_L) {
            itemMovementLocked = !itemMovementLocked;
            QString message = itemMovementLocked ? "Item movement locked." : "Item movement unlocked.";
            statusBar->showMessage(itemMovementLocked ? "Item movement locked." : "Item movement unlocked.", 2000);
        }
    }
}

void PackingViewer::recreateContainerGraphics(int pixelWidth) {
    // Create container
//    curMap->setVisible(false);
    QRectF containerPolygon = container->polygon().boundingRect();
    containerPolygon.setWidth((qreal)pixelWidth/this->rasterScale);
    container->setPolygon(containerPolygon);
    std::for_each(pieces.begin(), pieces.end(), [&containerPolygon](PackingItem *curItem){curItem->setContainer(containerPolygon);});
//    mainScene->setSceneRect(container->boundingRect());
}

void PackingViewer::createGraphicItems(RASTERPREPROCESSING::PackingProblem &problem) {
    // Create container
    std::shared_ptr<RASTERPREPROCESSING::Polygon> curContainer = (*problem.ccbegin())->getPolygon();
    QRectF containerPolygon = ((QPolygonF)(*curContainer)).boundingRect();
    container = new QGraphicsPolygonItem(containerPolygon);
    container->setBrush(QBrush(Qt::white));
    mainScene->addItem(container);

    // Create overlap map item
    curMap = new QGraphicsPixmapItem;
    curMap->setVisible(false);
    mainScene->addItem(curMap);

    // Create items
    int id = 0; int itemId = 0;
    for(QList<std::shared_ptr<RASTERPREPROCESSING::Piece>>::const_iterator it = problem.cpbegin(); it != problem.cpend(); it++, id++) {
        std::shared_ptr<RASTERPREPROCESSING::Polygon> curPiece = (*it)->getPolygon();
        for(uint mult = 0; mult < (*it)->getMultiplicity(); mult++, itemId++) {
            PackingItem *curStatic = new PackingItem(*curPiece, pieces.size());
            curStatic->setBrush(QColor(255,100,100,100));
            curStatic->setFlag(QGraphicsItem::ItemIsMovable, true);
            curStatic->setContainer(containerPolygon);
            std::for_each((*it)->corbegin(), (*it)->corend(), [&curStatic](int angle){curStatic->addAngle(angle);});
            pieces.append(curStatic);
            mainScene->addItem(curStatic);

            curStatic->setPos(curStatic->getInnerFitRectangle().topLeft());
        }
    }
    scale(1.0,-1.0);

    // Create origin circle
    originItem = new QGraphicsEllipseItem;
    originItem->setPen(QPen(Qt::black, 0));
    originItem->setBrush(Qt::blue);

    // Determine scale
    rasterScale = (*problem.crnfpbegin())->getScale(); // FIXME
    setZoomPosition(logposition(rasterScale));
    container->setPen(QPen(Qt::black, 1/rasterScale));
    std::for_each(pieces.begin(), pieces.end(), [this](PackingItem *curStatic){
        curStatic->setPen(QPen(Qt::red, 1/this->rasterScale));
        curStatic->setGridSize(1/this->rasterScale);
    });
    originItem->setRect(-3.5/rasterScale, -3.5/rasterScale,7.0/rasterScale,7.0/rasterScale);
    curMap->setScale(1/rasterScale);

    // Set initial configuration
    itemMovementLocked = false;
    setSelectedItem(0);
}

void PackingViewer::mouseMoveEvent ( QMouseEvent * event ) {
    if(!itemMovementLocked) {
        QGraphicsView::mouseMoveEvent(event);
        emit currentPositionChanged(pieces[this->currentPieceId]->pos());
    }
}

void PackingViewer::mousePressEvent(QMouseEvent * event) {
    if(event->button() == Qt::RightButton) setCurrentOrientation( (pieces[this->currentPieceId]->getCurAngle()+1)%pieces[this->currentPieceId]->getAnglesCount() );
            //pieces[this->currentPieceId]->rotateNext();

    if (itemAt(event->pos()) == NULL)
    {
      // do stuff if not clicked on an item
    }
    else
    {
        QGraphicsView::mousePressEvent(event);

        PackingItem* selItem;
        selItem = dynamic_cast<PackingItem*>(itemAt(event->pos()));

        if(selItem) {
            if(event->button() == Qt::LeftButton) setSelectedItem(selItem->getId());
        }
    }
}

void PackingViewer::setSelectedItem(int id) {
    if(this->currentPieceId != id) {
        if(this->currentPieceId != -1) {
            PackingItem *curStatic  = pieces[this->currentPieceId];
            curStatic->setPen(QPen(Qt::red, 1/rasterScale));
            curStatic->setBrush(QColor(255,100,100,100));
        }
        this->currentPieceId = id;
        PackingItem *curOrbiting = pieces[this->currentPieceId];
        curOrbiting->setPen(QPen(Qt::blue, 1/rasterScale));
        curOrbiting->setBrush(QColor(100,100,255,100));
        originItem->setParentItem(curOrbiting);
        curOrbiting->setZValue((*mainScene->items().begin())->zValue() + 0.1); // TEST

        emit selectedItemChanged(this->currentPieceId);
        emit currentPositionChanged(pieces[this->currentPieceId]->pos());
        curMap->setVisible(false);
    }
}

void PackingViewer::highlightPair(int id1, int id2) {
    highlightItem(id1); highlightItem(id2);
}

void PackingViewer::highlightItem(int id) {
    PackingItem *curItem  = pieces[id];
    curItem->setPen(QPen(QColor(255,0,255), 2.5/rasterScale));
    if(id == this->currentPieceId) originItem->setBrush(QColor(255,0,255));

    QTimer *timer = new QTimer(this);
    timer->setSingleShot(true);
    connect(timer, &QTimer::timeout, [curItem, id, this](){
        if(id == this->currentPieceId) {
            curItem->setPen(QPen(Qt::blue, 1/rasterScale));
            originItem->setBrush(Qt::blue);
        }
        else curItem->setPen(QPen(Qt::red, 1/rasterScale));
    });
    timer->start(2000);
}

void PackingViewer::setCurrentOrientation(int angle) {
    if(angle < 0) return;
    if(pieces[this->currentPieceId]->getCurAngle() != angle) {
        pieces[this->currentPieceId]->changeOrientation(angle);
        emit currentOrientationChanged(angle);
        curMap->setVisible(false);
    }

}

void PackingViewer::setCurrentXCoord(double xpos) {
    if(pieces[this->currentPieceId]->pos().x() != xpos) {
        pieces[this->currentPieceId]->setPos(xpos, pieces[this->currentPieceId]->pos().y());
        emit currentPositionChanged(pieces[this->currentPieceId]->pos());
    }
}


void PackingViewer::setCurrentYCoord(double ypos) {
    if(pieces[this->currentPieceId]->pos().y() != ypos) {
        pieces[this->currentPieceId]->setPos(pieces[this->currentPieceId]->pos().x(), ypos);
        emit currentPositionChanged(pieces[this->currentPieceId]->pos());
    }
}

void PackingViewer::getCurrentSolution(RASTERVORONOIPACKING::RasterPackingSolution &solution, qreal scale) {
    for(int i = 0; i < pieces.size(); i++) {
        solution.setOrientation(i, pieces[i]->getCurAngle());
        solution.setPosition(i, QPoint(qRound((qreal)pieces[i]->pos().x()*scale), qRound((qreal)pieces[i]->pos().y()*scale)));
    }
}

void PackingViewer::getCurrentSolution(RASTERVORONOIPACKING::RasterPackingSolution &solution) {
    getCurrentSolution(solution, this->rasterScale);
//    for(int i = 0; i < pieces.size(); i++) {
//        solution.setOrientation(i, pieces[i]->getCurAngle());
//        solution.setPosition(i, QPoint(qRound((qreal)pieces[i]->pos().x()*this->rasterScale), qRound((qreal)pieces[i]->pos().y()*this->rasterScale)));
//    }
}

void PackingViewer::setCurrentSolution(const RasterPackingSolution &solution, qreal scale) {
    for(int i = 0; i < solution.getNumItems(); i++) {
        pieces[i]->changeOrientation(solution.getOrientation(i));
        pieces[i]->setPos(QPointF((qreal)solution.getPosition(i).x()/scale, (qreal)solution.getPosition(i).y()/scale));
    }
    emit currentOrientationChanged(pieces[this->currentPieceId]->getCurAngle());
    emit currentPositionChanged(pieces[this->currentPieceId]->pos());
    curMap->setVisible(false);
}

void PackingViewer::setCurrentSolution(const RasterPackingSolution &solution) {
    setCurrentSolution(solution, this->rasterScale);
//    for(int i = 0; i < solution.getNumItems(); i++) {
//        pieces[i]->changeOrientation(solution.getOrientation(i));
//        pieces[i]->setPos(QPointF((qreal)solution.getPosition(i).x()/this->rasterScale, (qreal)solution.getPosition(i).y()/this->rasterScale));
//    }
//    emit currentOrientationChanged(pieces[this->currentPieceId]->getCurAngle());
//    emit currentPositionChanged(pieces[this->currentPieceId]->pos());
//    curMap->setVisible(false);
}

void PackingViewer::showTotalOverlapMap(std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> newMap) {
    // FIXME: Sometimes gives segmentation fault. Why???
    curMap->setPixmap(QPixmap::fromImage(newMap->getImage().copy()));
    curMap->setPos((1/rasterScale)*QPointF(-newMap->getReferencePoint().x()-0.5,-newMap->getReferencePoint().y()-0.5));
    curMap->setVisible(true);
}
