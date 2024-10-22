#include "runconfigurationsdialog.h"
#include "ui_runconfigurationsdialog.h"

RunConfigurationsDialog::RunConfigurationsDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::RunConfigurationsDialog)
{
    ui->setupUi(this);
    ui->spinBox->setValue(200);
	ui->label_7->setVisible(false); ui->spinBox_3->setVisible(false);
	ui->label_9->setVisible(false); ui->comboBox_4->setVisible(false);
	connect(ui->comboBox_2, SIGNAL(currentIndexChanged(int)), this, SLOT(showContainerLength(int)));
	connect(ui->comboBox_3, SIGNAL(currentIndexChanged(int)), this, SLOT(showRectangularMethods(int)));
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

int RunConfigurationsDialog::getZoomRatio() {
	return ui->spinBox_3->value();
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

void RunConfigurationsDialog::setInitialZoomFactor(int zoom) {
	ui->spinBox_3->setValue(zoom);
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

int RunConfigurationsDialog::getPackingProblemIndex() {
	return ui->comboBox_3->currentIndex();
}

bool RunConfigurationsDialog::isZoomedApproach() {
	return ui->checkBox->isChecked();
}

void RunConfigurationsDialog::showRectangularMethods(int index) {
	ui->comboBox_4->setVisible(index == 2);
	ui->label_9->setVisible(index == 2);
}

void RunConfigurationsDialog::showContainerLength(int index) {
	ui->doubleSpinBox->setEnabled(index == 1);
	ui->label_5->setEnabled(index == 1);
}

int RunConfigurationsDialog::getRectangularMethod() {
	return ui->comboBox_4->currentIndex();
}