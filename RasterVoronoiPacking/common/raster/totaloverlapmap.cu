#include "totaloverlapmap.h"
#include <stdio.h>
using namespace RASTERVORONOIPACKING;

__global__
void saxpy(int totalLines, int lineLength, int mapInitIdx, int nfpInitIdx, int mapOffsetHeight, int nfpOffsetHeight, quint32 *map, quint32 *nfp)
{
	int mapidx = mapInitIdx + blockIdx.x * (lineLength + mapOffsetHeight) + threadIdx.x;
	int nfpidx = nfpInitIdx + blockIdx.x * (lineLength + nfpOffsetHeight) + threadIdx.x;
	if (threadIdx.x < lineLength)
		map[mapidx] = nfp[nfpidx];
}

void TotalOverlapMap::addToMatrixCuda(int itemId, std::shared_ptr<quint32> d_nfp, QPoint nfporigin, int nfpwidth, int nfpheight, QPoint pos, std::shared_ptr<quint32> d_overlapMatrix, std::shared_ptr<quint32> d_overlapmap) {
	// Get intersection between innerfit and nofit polygon bounding boxes
	QPoint relativeOrigin = this->reference + pos - nfporigin;
	int relativeBotttomLeftX = relativeOrigin.x() < 0 ? -relativeOrigin.x() : 0;
	int relativeBotttomLeftY = relativeOrigin.y() < 0 ? -relativeOrigin.y() : 0;
	int relativeTopRightX = width - relativeOrigin.x(); relativeTopRightX = relativeTopRightX <  nfpwidth ? relativeTopRightX - 1 : nfpwidth - 1;
	int relativeTopRightY = height - relativeOrigin.y(); relativeTopRightY = relativeTopRightY < nfpheight ? relativeTopRightY - 1 : nfpheight - 1;

	// Create pointers to initial positions and calculate offsets for moving vertically
	int offsetHeight = height - (relativeTopRightY - relativeBotttomLeftY + 1);
	int nfpOffsetHeight = nfpheight - (relativeTopRightY - relativeBotttomLeftY + 1);
	int mapInitIdx = itemId * width * height + (relativeBotttomLeftX + relativeOrigin.x())*height + relativeBotttomLeftY + relativeOrigin.y();
	int nfpInitIdx = relativeBotttomLeftY + relativeBotttomLeftX * nfpheight;

	int totalLines = relativeTopRightX - relativeBotttomLeftX + 1;
	int lineLength = relativeTopRightY - relativeBotttomLeftY + 1;
	saxpy << < totalLines, 512 >> >(totalLines, lineLength, mapInitIdx, nfpInitIdx, offsetHeight, nfpOffsetHeight, d_overlapMatrix.get(), d_nfp.get());
	//cudaError_t e = cudaGetLastError();
	//if (e != cudaSuccess) printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));
	//cudaDeviceSynchronize();
}