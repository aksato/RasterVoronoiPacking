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
	qreal getSearchScale();
    int getMaxWorse();
    int getMaxSeconds();
    qreal getLenght();
	int getPackingProblemIndex();
	qreal getClusterFactor();
	bool isZoomedApproach();

    void setInitialLenght(qreal lenght, qreal step);
	void setInitialSearchScale(qreal scale);
	void enableCluster();
	void disableCluster();

private slots:

private:
    Ui::RunConfigurationsDialog *ui;
};

#endif // RUNCONFIGURATIONSDIALOG_H
