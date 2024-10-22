#include "glsweightset.h"
#include "rasterpackingproblem.h"
#include <QImage>
#include <QDebug>
#include "colormap.h"

#define INITWEIGHTVAL 100
using namespace RASTERVORONOIPACKING;

void GlsWeightSet::init(int numItems) {
	weights = QVector<unsigned int>(numItems * numItems);
	weights.fill(INITWEIGHTVAL);
}

int GlsWeightSet::getWeight(int itemId1, int itemId2) {
    Q_ASSERT_X(itemId1 != itemId2, "GlsWeightSet::getWeigth", "Cannot verify collision with itself");
	if (itemId1 > itemId2) return weights[itemId1 + numItems*itemId2];
	return weights[itemId2 + numItems*itemId1];
}

void GlsWeightSet::addWeight(int itemId1, int itemId2, int weight) {
    Q_ASSERT_X(itemId1 != itemId2, "GlsWeightSet::getWeigth", "Cannot verify collision with itself");
	if (itemId1 > itemId2) { weights[itemId1 + numItems*itemId2] = weight; return; }
	weights[itemId2 + numItems*itemId1] = weight;
}

void GlsWeightSet::updateWeights(QVector<WeightIncrement> &increments) {
    std::for_each(increments.begin(), increments.end(), [this](WeightIncrement &inc){
		if (inc.id1 > inc.id2) weights[inc.id1 + numItems*inc.id2] += inc.value;
		else weights[inc.id2 + numItems*inc.id1] += inc.value;
    });
}

QImage GlsWeightSet::getImage(int numItems) {
    QImage image(numItems, numItems, QImage::Format_Indexed8);
	image.fill(0);
    setColormap(image, false);

	qreal minW = std::numeric_limits<quint32>::max();
    qreal maxW = 1.0;

    for(int itemId1 = 0; itemId1 < numItems; itemId1++)
        for(int itemId2 = 0; itemId2 < numItems; itemId2++)
            if(itemId1 != itemId2) {
                qreal curW = getWeight(itemId1, itemId2);
                if(curW < minW) minW = curW;
                if(curW > maxW) maxW = curW;
            }

	for (int itemId2 = 0; itemId2 < numItems; itemId2++) {
		uchar *imageLine = (uchar *)image.scanLine(itemId2);
		for (int itemId1 = 0; itemId1 < numItems; itemId1++, imageLine++) {
			if (itemId1 != itemId2) {
				qreal curW = getWeight(itemId1, itemId2);
				int index = qRound(255.0 * (curW - minW) / (maxW - minW));
				//image.setPixel(itemId1, itemId2, index);
				*imageLine = index;
			}
		}
	}
            //else image.setPixel(itemId1, itemId2, 0);
    return image;
}

void GlsWeightSet::reset(int numItems) {
	weights.fill(INITWEIGHTVAL);
}
