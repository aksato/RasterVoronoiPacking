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

void ZoomedMapViewDialog::setImage(QPixmap _pmap) {
    ui->graphicsView->setImage(_pmap);
}

void ZoomedMapViewDialog::setValidArea(QRect validArea) {
    ui->graphicsView->setValidArea(validArea);
}
