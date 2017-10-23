#include "zoomedmapviewdialog.h"
#include "ui_zoomedmapviewdialog.h"

ZoomedMapViewDialog::ZoomedMapViewDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ZoomedMapViewDialog)
{
    ui->setupUi(this);
}

ZoomedMapViewDialog::~ZoomedMapViewDialog()
{
    delete ui;
}

ZoomedMapView *ZoomedMapViewDialog::getMapView() {
	return ui->graphicsView;
}

//void ZoomedMapViewDialog::init(int size) {
//	ui->graphicsView->init(size);
//}

//void ZoomedMapViewDialog::updateMap(std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> map) {
//	ui->graphicsView->updateMap(map);
//}

void ZoomedMapViewDialog::setValidArea(QRect validArea) {
    ui->graphicsView->setValidArea(validArea);
}
