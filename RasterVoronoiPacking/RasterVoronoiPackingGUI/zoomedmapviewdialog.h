#ifndef ZOOMEDMAPVIEWDIALOG_H
#define ZOOMEDMAPVIEWDIALOG_H

#include <QDialog>
#include <QPixmap>

namespace Ui {
class ZoomedMapViewDialog;
}

class ZoomedMapViewDialog : public QDialog
{
    Q_OBJECT

public:
    explicit ZoomedMapViewDialog(QWidget *parent = 0);
    ~ZoomedMapViewDialog();

    void setImage(QPixmap _pmap);
    void setValidArea(QRect validArea);

private:
    Ui::ZoomedMapViewDialog *ui;
};

#endif // ZOOMEDMAPVIEWDIALOG_H
