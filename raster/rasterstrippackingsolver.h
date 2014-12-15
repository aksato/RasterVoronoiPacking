#ifndef RASTERSTRIPPACKINGSOLVER_H
#define RASTERSTRIPPACKINGSOLVER_H
#include <memory>
#include "rasterpackingproblem.h"
#include "rasterpackingsolution.h"
#include "totaloverlapmap.h"
#include "glsweightset.h"

namespace RASTERVORONOIPACKING {

    class RasterStripPackingSolver
    {
    public:
        RasterStripPackingSolver() {}
        RasterStripPackingSolver(std::shared_ptr<RasterPackingProblem> _problem) {setProblem(_problem);}
        void setProblem(std::shared_ptr<RasterPackingProblem> _problem, bool isZoomedProblem = false);

        // Info Functions
        int getNumItems() {return originalProblem->count();}
        int getCurrentWidth() {return currentWidth;}
        std::shared_ptr<GlsWeightSet> getGlsWeights() {return glsWeights;}

        // Basic Functions
        void generateRandomSolution(RasterPackingSolution &solution);
        // --> Get layout overlap (sum of individual overlap values)
        qreal getGlobalOverlap(RasterPackingSolution &solution);
        // --> Local search
        void performLocalSearch(RasterPackingSolution &solution, bool useGlsWeights = false);
        // --> Switch between original and zoomed problems
        void switchProblem(bool zoomedProblem);
        // --> Local search with zoomed approach
        void performTwoLevelLocalSearch(RasterPackingSolution &zoomedSolution,bool useGlsWeights = false, int neighboordScale = 1);
        // --> GLS weights functions
        void updateWeights(RasterPackingSolution &solution);
        void resetWeights();

        // To be private functions
        // --> Return total overlap map for a given item
        std::shared_ptr<TotalOverlapMap> getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution, bool useGlsWeights = false);
        // --> Get absolute minimum overlap position
        QPoint getMinimumOverlapPosition(std::shared_ptr<TotalOverlapMap> map, qreal &value);
        // --> Retrieve a rectangular area of the total overlap map
        std::shared_ptr<TotalOverlapMap> getRectTotalOverlapMap(int itemId, int orientation, QPoint pos, int width, int height, RasterPackingSolution &solution, bool useGlsWeights = false);

        // Auxiliary functions
        // --> Change container size
        void setContainerWidth(int pixelWidth);

        // Debug only functions
        // --> Get layout overlap with individual values (sum of individual overlap values)
        qreal getGlobalOverlap(RasterPackingSolution &solution, QVector<qreal> &individualOverlaps);

    private:
        // --> Get two items overlap
        qreal getDistanceValue(int itemId1, QPoint pos1, int orientation1, int itemId2, QPoint pos2, int orientation2);
        qreal getItemTotalOverlap(int itemId, RasterPackingSolution &solution);
        // TEST
        qreal getTotalOverlapMapSingleValue(int itemId, int orientation, QPoint pos, RasterPackingSolution &solution, bool useGlsWeights = false);

        std::shared_ptr<RasterPackingProblem> currentProblem;
        std::shared_ptr<RasterPackingProblem> originalProblem;
        std::shared_ptr<RasterPackingProblem> zoomedProblem;
        TotalOverlapMapSet maps;
        std::shared_ptr<GlsWeightSet> glsWeights;
        int currentWidth, initialWidth;
    };

}

#endif // RASTERSTRIPPACKINGSOLVER_H
