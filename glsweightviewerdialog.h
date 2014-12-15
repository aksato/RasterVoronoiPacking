#ifndef GLSWEIGHTVIEWERDIALOG_H
#define GLSWEIGHTVIEWERDIALOG_H

#include <QDialog>
#include <Qimage>
#include <QGraphicsPixmapItem>
#include <memory>
#include "raster/glsweightset.h"
#include "viewer/weightviewer.h"

namespace Ui {
class GlsWeightViewerDialog;
}

class GlsWeightViewerDialog : public QDialog
{
    Q_OBJECT

public:
    explicit GlsWeightViewerDialog(QWidget *parent = 0);
    ~GlsWeightViewerDialog();

    void setWeights(std::shared_ptr<RASTERVORONOIPACKING::GlsWeightSet> w, int numItems);

signals:
    void weightViewerSelectionChanged(int i, int j);

public slots:
    void updateWeightViewerSelection(int i, int j) {emit weightViewerSelectionChanged(i,j);}
    void updateImage();

private:
    Ui::GlsWeightViewerDialog *ui;
};

#endif // GLSWEIGHTVIEWERDIALOG_H
