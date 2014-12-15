#ifndef GLSWEIGHTSET_H
#define GLSWEIGHTSET_H

#include <memory>
#include <QtGlobal>
#include <QHash>
#include <QVector>

class QImage;

namespace RASTERVORONOIPACKING {
    class RasterPackingSolution;
    class RasterPackingProblem;

    struct WeightIncrement {
        WeightIncrement() {}
        WeightIncrement(int _id1, int _id2, qreal _value) {
            id1 = _id1; id2 = _id2; value = _value;
        }
        int id1, id2;
        qreal value;
    };

    class GlsWeightSet
    {
    public:
        GlsWeightSet() {}
        GlsWeightSet(int numItems) {init(numItems);}

        void init(int numItems);
        void clear() {weights.clear();}
        void reset(int numItems);
        qreal getWeight(int itemId1, int itemId2);
        void updateWeights(QVector<WeightIncrement> &increments);

        QImage getImage(int numItems);

    private:
        void addWeight(int itemId1, int itemId2, qreal weight);

        QHash<QPair<int,int>, qreal> weights;

    };
}

#endif // GLSWEIGHTSET_H
