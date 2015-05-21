#ifndef GPUINFO_H
#define GPUINFO_H

namespace CUDAPACKING {

	struct cuPoint {
		cuPoint(){}
		cuPoint(int _x, int _y){ x = _x; y = _y; }
		int x, y;
	};

	class CudaRasterNoFitPolygon {
	public:
		CudaRasterNoFitPolygon(int *_matrix, cuPoint _origin, float _maxD) : matrix(_matrix), origin(_origin), maxD(_maxD) {}
		~CudaRasterNoFitPolygon() {}

		void setOrigin(cuPoint origin) { this->origin = origin; }
		cuPoint getOrigin() { return this->origin; }
		int getOriginX() { return this->origin.x; }
		int getOriginY() { return this->origin.y; }
		float getMaxD() { return this->maxD; }

		int getPixel(int i, int j) { return matrix[j*m_width + i]; }
		int *getMatrix() { return matrix; }
		void setMatrix(int *_matrix) { matrix = _matrix; }
		int width() { return m_width; }
		int height() { return m_height; }
		void setWidth(int _m_width) { m_width = _m_width; }
		void setHeight(int _m_height) { m_height = _m_height; }

		cuPoint origin;
		int m_width, m_height;
		int *matrix;
	private:

		float maxD;


	};

	bool getTotalMemory(int &gpuDeviceCount, size_t &free, size_t &total);
	void allocItemTypes(int numItems);
	void alloDevicecSolutionPointers(int numItems);
	void setItemType(int itemId, int typeId);
	void allocHostNfpPointers(int numItems, int numOrientations);
	void allocSingleDeviceNfpMatrix(int staticId, int orbitingId, int *matrix, int width, int height, int originx, int originy);
	void allocDeviceNfpPointers(int numItems, int numOrientations);
	void allocDeviceMaxIfp(size_t memSize);
	bool reallocDeviceMaxIfp(size_t memSize);
	float *getcuOverlapMap(int curItem, int curItemAngle, int nItems, int numAngles, int overlapmap_width, int overlapmap_height, int overlapmapx, int overlapmapy, int *posx, int *posy, int *angles, float *weights, bool useGlsWeights = false);
	float *getcuOverlapMap2(int curItem, int curItemAngle, int nItems, int numAngles, int overlapmap_width, int overlapmap_height, int overlapmapx, int overlapmapy, int *posx, int *posy, int *angles, float *weights, bool useGlsWeights = true);
	float getcuMinimumOverlap(int curItem, int curItemAngle, int nItems, int numAngles, int overlapmap_width, int overlapmap_height, int overlapmapx, int overlapmapy, int *posx, int *posy, int *angles, float *weights, int &minX, int &minY, bool useGlsWeights = false);
}

#endif // GPUINFO_H