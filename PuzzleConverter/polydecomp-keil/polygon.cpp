#include <algorithm>
#include <limits>
#include "polygon.h"
#include "line.h"

using namespace POLYDECOMP;

Point& Polygon::operator[](const int &i) {
    return v[i];
}

vector<Point>::iterator Polygon::begin() {
	return v.begin();
}

vector<Point>::iterator Polygon::end() {
	return v.end();
}

Point& Polygon::at(const int &i) {
    const int s = v.size();
    return v[i < 0 ? i % s + s : i % s];
}

Point& Polygon::first() {
    return v.front();
}

Point& Polygon::last() {
    return v.back();
}

int Polygon::size() const {
    return v.size();
}

void Polygon::push(const Point &p) {
    v.push_back(p);
}

void Polygon::clear() {
    v.clear();
    tp.clear();
    tl.clear();
}

void Polygon::makeCCW() {
    int br = 0;

    // find bottom right point
    for (int i = 1; i < v.size(); ++i) {
        if (v[i].y < v[br].y || (v[i].y == v[br].y && v[i].x > v[br].x)) {
            br = i;
        }
    }

    // reverse poly if clockwise
    if (!left(at(br - 1), at(br), at(br + 1))) {
        reverse();
    }
}

void Polygon::reverse() {
    std::reverse(v.begin(), v.end());
}

bool Polygon::isReflex(const int &i) {
    return right(at(i - 1), at(i), at(i + 1));
}

bool Polygon::canSee(const int &a, const int &b) {
    Point p;
    Scalar dist;

    if (leftOn(at(a + 1), at(a), at(b)) && rightOn(at(a - 1), at(a), at(b))) {
        return false;
    }
    dist = sqdist(at(a), at(b));
    for (int i = 0; i < v.size(); ++i) { // for each edge
        if ((i + 1) % v.size() == a || i == a) // ignore incident edges
            continue;
        if (leftOn(at(a), at(b), at(i + 1)) && rightOn(at(a), at(b), at(i))) { // if diag intersects an edge
            p = lineInt(Line(at(a), at(b)), Line(at(i), at(i + 1)));
            if (sqdist(at(a), p) < dist) { // if edge is blocking visibility to b
                return false;
            }
        }
    }

    return true;
}

Polygon Polygon::copy(const int &i, const int &j) {
    Polygon p;
    if (i < j) {
        p.v.insert(p.v.begin(), v.begin() + i, v.begin() + j + 1);
    } else {
        p.v.insert(p.v.begin(), v.begin() + i, v.end());
        p.v.insert(p.v.end(), v.begin(), v.begin() + j + 1);
    }
    return p;
}

EdgeList Polygon::decomp() {
    EdgeList min, tmp1, tmp2;
    int nDiags = std::numeric_limits<int>::max();

    for (int i = 0; i < v.size(); ++i) {
        if (isReflex(i)) {
            for (int j = 0; j < v.size(); ++j) {
                if (canSee(i, j)) {
                    tmp1 = copy(i, j).decomp();
                    tmp2 = copy(j, i).decomp();
                    tmp1.insert(tmp1.end(), tmp2.begin(), tmp2.end());
                    if (tmp1.size() < nDiags) {
                        min = tmp1;
                        nDiags = tmp1.size();
                        min.push_back(Edge(at(i), at(j)));
                    }
                }
            }
        }
    }
    
    return min;
}

/**
* Slices the polygon given one or more cut edges. If given one, this function will return two polygons (false on failure). If many, an array of polygons.
* @method slice
* @param {Array} cutEdges A list of edges, as returned by .getCutEdges()
* @return {Array}
*/
std::vector<Polygon> Polygon::slice(Polygon &polygon, EdgeList &cutEdges){
	if (cutEdges.size() == 0){
		std::vector<Polygon> polys;
		polys.push_back(polygon);
		return polys;
	}
	if (cutEdges.size() > 1) {// instanceof Array && cutEdges.length && cutEdges[0] instanceof Array && cutEdges[0].length == = 2 && cutEdges[0][0] instanceof Array){

		std::vector<Polygon> polys;
		polys.push_back(polygon);

		for (int i = 0; i<cutEdges.size(); i++){
			EdgeList cutEdge;
			cutEdge.push_back(cutEdges[i]);
			// Cut all polys
			for (int j = 0; j<polys.size(); j++){
				Polygon poly = polys[j];
				std::vector<Polygon> result = slice(poly, cutEdge);
				if (result.size() > 0){
					// Found poly! Cut and quit
					polys.erase(polys.begin() + j); //polys.splice(j, 1);
					polys.push_back(result[0]);
					polys.push_back(result[1]);
					break;
				}
			}
		}

		return polys;
	}
	else {

		// Was given one edge
		Edge cutEdge = cutEdges[0];
		auto iti = std::find(polygon.begin(), polygon.end(), cutEdge.first); //int i = polygon.indexOf(cutEdge.first);
		auto itj = std::find(polygon.begin(), polygon.end(), cutEdge.second); //int j = polygon.indexOf(cutEdge.second);
		
		if (iti != polygon.end() && itj != polygon.end()){ //if (i != = -1 && j != = -1){
			int i = std::distance(polygon.begin(), iti);
			int j = std::distance(polygon.begin(), itj);

			std::vector<Polygon> polys;
			polys.push_back(polygon.copy(i, j));
			polys.push_back(polygon.copy(j, i));
			return polys;
			//return[polygonCopy(polygon, i, j),
			//	polygonCopy(polygon, j, i)];
		}
		else {
			std::vector<Polygon> polys;
			return polys;
		}
	}
}

std::vector<Polygon> Polygon::sliceInTwo(Polygon &polygon, Edge &cutEdge) {
	// Was given one edge
	auto iti = std::find(polygon.begin(), polygon.end(), cutEdge.first); //int i = polygon.indexOf(cutEdge.first);
	auto itj = std::find(polygon.begin(), polygon.end(), cutEdge.second); //int j = polygon.indexOf(cutEdge.second);

	if (iti != polygon.end() && itj != polygon.end()){ //if (i != = -1 && j != = -1){
		int i = std::distance(polygon.begin(), iti);
		int j = std::distance(polygon.begin(), itj);

		std::vector<Polygon> polys;
		polys.push_back(polygon.copy(i, j));
		polys.push_back(polygon.copy(j, i));
		//return[polygonCopy(polygon, i, j),
		//	polygonCopy(polygon, j, i)];
	}
	else {
		std::vector<Polygon> polys;
		return polys;
	}
}