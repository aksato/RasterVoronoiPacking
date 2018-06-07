#ifndef _POLYGON_H
#define	_POLYGON_H

#include <vector>
#include "config.h"
#include "point.h"
#include "line.h"
#include "edge.h"

namespace POLYDECOMP {
	class Polygon {
	public:
		Point & operator[](const int &i);
		Point& at(const int &i);
		Point& first();
		Point& last();
		int size() const;
		EdgeList decomp();
		bool isReflex(const int &i);
		bool canSee(const int &a, const int &b);
		void push(const Point &p);
		void clear();
		void makeCCW();
		void reverse();
		Polygon copy(const int &i, const int &j);
		std::vector<Polygon> slice(Polygon &polygon, EdgeList &cutEdges);

		vector<Point>::iterator begin();
		vector<Point>::iterator end();
		vector<Point> tp;
		vector<Line> tl;
	private:
		vector<Point> v;
		static std::vector<Polygon> sliceInTwo(Polygon &polygon, Edge &cutEdge);
	};
}
#endif	/* _POLYGON_H */

