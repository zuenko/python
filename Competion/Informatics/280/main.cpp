#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
 
struct tPoint {
    int x, y;
};
 
struct tSegment {
    tPoint pt1, pt2;
};
 
bool findPt (const tSegment, const tSegment); // ищет общую точку двух отрезков.
bool oneLine (const tSegment, const tSegment); // лежат ли отрезки на одной прямой.
bool commonPart (const tSegment, const tSegment); // имеется ли общая часть у отрезков.
int side (const tSegment, const tPoint); // размещение точки относительно прямой.
int down (const int); // уменьшает число, сохраняя знак.
 
int main() {
    std::ifstream ifs ("input.txt");
    std::ofstream ofs ("output.txt");
    //
    tSegment first, second;
    ifs >> first.pt1.x >> first.pt1.y;
    ifs >> first.pt2.x >> first.pt2.y;
    ifs >> second.pt1.x >> second.pt1.y;
    ifs >> second.pt2.x >> second.pt2.y;
    //
    ofs << (findPt (first, second) ? "Yes" : "No");
    //
    ifs.close();
    ofs.close();
    return 0;
}
 
bool findPt (const tSegment first, const tSegment second) {
    if (oneLine (first, second)) { // если прямые на одной прямой, смотрим,
        if (commonPart (first, second)) // имеют ли они общую часть.
            return true;
        return false;
    }
 
    int rez1 = side (first, second.pt1);
    int rez2 = side (first, second.pt2);
    int rez3 = side (second, first.pt1);
    int rez4 = side (second, first.pt2);
 
    if ((!rez1) || (!rez2) || (!rez3) || (!rez4))
        return true;
 
    return ((rez1 * rez2 < 0) && (rez3 * rez4 < 0));
}
 
bool oneLine (const tSegment first, const tSegment second) {
    return ((!side (first, second.pt1)) && (!side (first, second.pt2)));
}
 
bool commonPart (const tSegment first, const tSegment second) {
    int minx1 = std::min (first.pt1.x, first.pt2.x);
    int miny1 = std::min (first.pt1.y, first.pt2.y);
    int maxx1 = std::max (first.pt1.x, first.pt2.x);
    int maxy1 = std::max (first.pt1.y, first.pt2.y);
    int minx2 = std::min (second.pt1.x, second.pt2.x);
    int miny2 = std::min (second.pt1.y, second.pt2.y);
    int maxx2 = std::max (second.pt1.x, second.pt2.x);
    int maxy2 = std::max (second.pt1.y, second.pt2.y);
 
    return (!((minx1 > maxx2) || (maxx1 < minx2) ||
              (miny1 > maxy2) || (maxy1 < miny2)));
}
 
int side (const tSegment segment, const tPoint pt) {
    return down (((pt.x - segment.pt1.x) * (segment.pt2.y - segment.pt1.y) -
            (pt.y - segment.pt1.y) * (segment.pt2.x - segment.pt1.x)));
}
 
int down (const int x) {
    return (x > 0 ? 1 : x == 0 ? 0 : -1);
}
