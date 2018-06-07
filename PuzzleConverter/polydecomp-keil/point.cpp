#include "point.h"

using namespace POLYDECOMP;

Point::Point() : x(0), y(0) {
}

Point::Point(Scalar x, Scalar y) : x(x), y(y) {
}

Scalar& Point::operator [](int i) {
    return v[i];
}

bool Point::operator == (const Point &p) {
	return x == p.x && y == p.y;
}

Scalar POLYDECOMP::area(const Point &a, const Point &b, const Point &c) {
    return (((b.x - a.x)*(c.y - a.y))-((c.x - a.x)*(b.y - a.y)));
}

bool POLYDECOMP::left(const Point &a, const Point &b, const Point &c) {
    return POLYDECOMP::area(a, b, c) > 0;
}

bool POLYDECOMP::leftOn(const Point &a, const Point &b, const Point &c) {
    return POLYDECOMP::area(a, b, c) >= 0;
}

bool POLYDECOMP::right(const Point &a, const Point &b, const Point &c) {
    return POLYDECOMP::area(a, b, c) < 0;
}

bool POLYDECOMP::rightOn(const Point &a, const Point &b, const Point &c) {
    return POLYDECOMP::area(a, b, c) <= 0;
}

bool POLYDECOMP::collinear(const Point &a, const Point &b, const Point &c) {
    return POLYDECOMP::area(a, b, c) == 0;
}

Scalar POLYDECOMP::sqdist(const Point &a, const Point &b) {
    Scalar dx = b.x - a.x;
    Scalar dy = b.y - a.y;
    return dx * dx + dy * dy;
}