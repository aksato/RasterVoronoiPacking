#include "rasterpackingsolution.h"

using namespace RASTERVORONOIPACKING;

RasterPackingSolution::RasterPackingSolution()
{
}

RasterPackingSolution::RasterPackingSolution(int numItems)
{
    for(int i = 0; i < numItems; i++)
        placements.append(RasterItemPlacement());
}

QDebug operator<<(QDebug dbg, const RasterPackingSolution &c)
{
    for(int i = 0; i < c.getNumItems(); i++) {
        dbg.nospace() << "[Item:" << i << ", Pos:(" << c.getPosition(i).x() << "," << c.getPosition(i).y() << "), Angle:" << c.getOrientation(i) << "]\n";
    }

    return dbg.space();
}
