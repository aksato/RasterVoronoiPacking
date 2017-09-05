#include "runconfigurationsdialog.h"
#include "ui_runconfigurationsdialog.h"

RunConfigurationsDialog::RunConfigurationsDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::RunConfigurationsDialog)
{
    ui->setupUi(this);
    ui->spinBox->setValue(200);
}

RunConfigurationsDialog::~RunConfigurationsDialog()
{
    delete ui;
}

int RunConfigurationsDialog::getInitialSolution() {
    return ui->comboBox_2->currentIndex();
}

int RunConfigurationsDialog::getMetaheuristic() {
    return ui->comboBox->currentIndex();
}

int RunConfigurationsDialog::getMaxWorse() {
    return ui->spinBox->value();
}

int RunConfigurationsDialog::getMaxSeconds() {
    return QTime(0,0).secsTo(ui->timeEdit->time());
}

qreal RunConfigurationsDialog::getLenght() {
    return ui->doubleSpinBox->value();
}

void RunConfigurationsDialog::setInitialLenght(qreal lenght, qreal step) {
    ui->doubleSpinBox->setValue(lenght);
    ui->doubleSpinBox->setMinimum(0.0);
    //ui->doubleSpinBox->setMaximum(lenght);
    ui->doubleSpinBox->setSingleStep(step);
}

bool RunConfigurationsDialog::getStripPacking() {
	return ui->checkBox->isChecked();
}

qreal RunConfigurationsDialog::getClusterFactor() {
	if (!ui->doubleSpinBox_2->isEnabled()) return -1.0;
	return ui->doubleSpinBox_2->value();
}

void RunConfigurationsDialog::enableCluster() {
	ui->label_6->setEnabled(true);
	ui->doubleSpinBox_2->setEnabled(true);
}

void RunConfigurationsDialog::disableCluster() {
	ui->label_6->setEnabled(false);
	ui->doubleSpinBox_2->setEnabled(false);
}