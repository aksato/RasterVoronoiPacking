#ifndef RUNCONFIGURATIONSDIALOG_H
#define RUNCONFIGURATIONSDIALOG_H

#include <QDialog>

namespace Ui {
class RunConfigurationsDialog;
}

class RunConfigurationsDialog : public QDialog
{
    Q_OBJECT

public:
    explicit RunConfigurationsDialog(QWidget *parent = 0);
    ~RunConfigurationsDialog();

    int getInitialSolution();
    int getMetaheuristic();
	int getZoomRatio();
    int getMaxWorse();
    int getMaxSeconds();
    qreal getLenght();
	int getPackingProblemIndex();
	int getRectangularMethod();
	qreal getClusterFactor();
	bool isZoomedApproach();

    void setInitialLenght(qreal lenght, qreal step);
	void setInitialZoomFactor(int zoom);
	void enableCluster();
	void disableCluster();

private slots:
	void showRectangularMethods(int index);
	void showContainerLength(int index);
private:
    Ui::RunConfigurationsDialog *ui;
};

#endif // RUNCONFIGURATIONSDIALOG_H
