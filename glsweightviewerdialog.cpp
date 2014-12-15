#include "glsweightviewerdialog.h"
#include "ui_glsweightviewerdialog.h"

GlsWeightViewerDialog::GlsWeightViewerDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::GlsWeightViewerDialog)
{
    ui->setupUi(this);

    connect(ui->graphicsView, SIGNAL(selectionChanged(int,int)), this, SLOT(updateWeightViewerSelection(int,int)));
}

GlsWeightViewerDialog::~GlsWeightViewerDialog()
{
    delete ui;
}

void GlsWeightViewerDialog::setWeights(std::shared_ptr<RASTERVORONOIPACKING::GlsWeightSet> w, int numItems) {
    ui->graphicsView->init(numItems);
    ui->graphicsView->setWeights(w);
}

void GlsWeightViewerDialog::updateImage() {
    ui->graphicsView->updateImage();

}
