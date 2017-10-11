#ifndef ZOOMEDMAPVIEWDIALOG_H
#define ZOOMEDMAPVIEWDIALOG_H

#include <QDialog>
#include <QPixmap>
#include "../common/raster/totaloverlapmap.h"
#include <memory>
#include "viewer/zoomedmapview.h"

namespace Ui {
class ZoomedMapViewDialog;
}

class ZoomedMapViewDialog : public QDialog
{
    Q_OBJECT

public:
    explicit ZoomedMapViewDialog(QWidget *parent = 0);
    ~ZoomedMapViewDialog();

	ZoomedMapView *getMapView();
    //void init(int size);
	//void updateMap(std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> map);
    void setValidArea(QRect validArea);

private:
    Ui::ZoomedMapViewDialog *ui;
};

#endif // ZOOMEDMAPVIEWDIALOG_H
