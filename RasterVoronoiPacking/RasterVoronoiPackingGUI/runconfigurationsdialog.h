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
    int getMaxWorse();
    int getMaxSeconds();
    qreal getLenght();
	bool getUseCUDA();

    void setInitialLenght(qreal lenght, qreal step);
private:
    Ui::RunConfigurationsDialog *ui;
};

#endif // RUNCONFIGURATIONSDIALOG_H
