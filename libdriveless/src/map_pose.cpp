#include "../include/map_pose.h"
#include <cmath>
#include <memory>
#include <stdio.h>

MapPose MapPose::clone()
{
    return MapPose(
        _x,
        _y,
        _z,
        _heading);
}

bool MapPose::areClose_2D(MapPose &p1, MapPose &p2)
{
    return __TOLERANCE_CLOSE_VALUE(p1.x(), p2.x()) &&
           __TOLERANCE_CLOSE_VALUE(p1.y(), p2.y());
}
bool MapPose::areClose_3D(MapPose &p1, MapPose &p2)
{
    return __TOLERANCE_CLOSE_VALUE(p1.x(), p2.x()) &&
           __TOLERANCE_CLOSE_VALUE(p1.y(), p2.y()) &&
           __TOLERANCE_CLOSE_VALUE(p1.z(), p2.z());
}

double MapPose::DistanceBetween_2D(MapPose &p1, MapPose &p2)
{
    double dx = p2.x() - p1.x();
    double dy = p2.y() - p1.y();
    return sqrt(dx * dx + dy * dy);
}

double MapPose::DistanceBetween_3D(MapPose &p1, MapPose &p2)
{
    double dx = p2.x() - p1.x();
    double dy = p2.y() - p1.y();
    double dz = p2.z() - p1.z();
    return sqrt(dx * dx + dy * dy + dz * dz);
}

double MapPose::dot_2D(MapPose &p1, MapPose &p2)
{
    return p1.x() * p2.x() + p1.y() * p2.y();
}

double MapPose::dot_3D(MapPose &p1, MapPose &p2)
{
    return p1.x() * p2.x() + p1.y() * p2.y() + p1.z() * p2.z();
}

double MapPose::distanceToLine_2D(
    double line_x1,
    double line_y1,
    double line_x2,
    double line_y2,
    double x,
    double y)
{
    double dx = line_x2 - line_x1;
    double dy = line_y2 - line_y1;

    if (dx == 0 && dy == 0)
        return 0;

    double num = dx * (line_y1 - y) - (line_x1 - x) * dy;
    double den = sqrt(dx * dx + dy * dy);
    return num / den;
}

double MapPose::distanceToLine_2D(MapPose &lineP1, MapPose &lineP2, MapPose &p)
{
    return distanceToLine_2D(lineP1.x(), lineP1.y(), lineP2.x(), lineP2.y(), p.x(), p.y());
}

angle MapPose::computePathHeading_2D(MapPose &p1, MapPose &p2)
{
    double angle = computePathHeading_2D_rad(p1.x(), p1.y(), p2.x(), p2.y());
    return angle::rad(angle);
}

double MapPose::computePathHeading_2D_rad(double line_x1,
                                          double line_y1,
                                          double line_x2,
                                          double line_y2)
{
    double dx = line_x2 - line_x1;
    double dy = line_y2 - line_y1;

    if (dy >= 0 && dx > 0) // Q1
        return atan(dy / dx);
    else if (dy >= 0 && dx < 0) // Q2
        return -atan(dy / abs(dx));
    else if (dy < 0 && dx > 0) // Q3
        return -atan(abs(dy) / dx);
    else if (dy < 0 && dx < 0) // Q4
        return atan(dy / dx) - PI;
    else if (dx == 0 && dy > 0)
        return HALF_PI;
    else if (dx == 0 && dy < 0)
        return -HALF_PI;
    return 0.0;
}

/* V2
std::unique_ptr<PathProjection> MapPose::projectOnPath(MapPose &p1, MapPose &p2, MapPose &p)
{
    // Direction vector
    double dx = p2.x() - p1.x();
    double dy = p2.y() - p1.y();

    double pathSizeSqd = dx * dx + dy * dy;

    if (pathSizeSqd == 0)
        return nullptr;

    // Vector from (x1, y1) to (x, y)
    double vx = p.x() - p1.x();
    double vy = p.y() - p1.y();

    //Scalar projection (distance along the arrow direction)
    double dot_product = vx * dx + vy * dy;
    double t = dot_product / pathSizeSqd;
      
    PathProjection *proj = new PathProjection{
        MapPose(
            p1.x() + t * dx,
            p1.y() + t * dy,
            0, angle::rad(0)),
        0,
        sqrt(pathSizeSqd)};

    proj->distanceFromP1 = MapPose::DistanceBetween_2D(proj->pose, p1);

    return std::unique_ptr<PathProjection>(proj);
} */

std::unique_ptr<PathProjection> MapPose::projectOnPath(MapPose &p1, MapPose &p2, MapPose &p)
{
    double pathSize = MapPose::DistanceBetween_2D(p1, p2);

    if (pathSize == 0)
        return nullptr;

    MapPose l = MapPose(
        (p2.x() - p1.x()) / pathSize,
        (p2.y() - p1.y()) / pathSize,
        0, angle::rad(0));
    MapPose v = MapPose(
        (p._x - p1.x()),
        (p._y - p1.y()),
        0, angle::rad(0));

    double distanceFromP1 = MapPose::dot_2D(v, l);

    PathProjection *proj = new PathProjection{
        MapPose(
            p1.x() + l._x * distanceFromP1,
            p1.y() + l._y * distanceFromP1,
            0, angle::rad(0)),
        distanceFromP1,
        pathSize};

    return std::unique_ptr<PathProjection>(proj);
}

inline double MapPose::squaredDistanceToMidLine(MapPose &p1, MapPose &p2, MapPose &p) {
    double dx = (0.5*(p2.x() + p1.x()))-p.x();
    double dy = (0.5*(p2.y() + p1.y()))-p.y();
    return dx * dx + dy * dy;
}

NearestPoseSearchResult MapPose::findNearestPosePosOnList(std::vector<MapPose> &list, MapPose &location, int firstPositionOnList, int maxHopping)
{
    double bestDistance = 999999999;
    int bestPos = -1;
    int lastPositionOnList = list.size() - 1;
    NearestPoseType bestPostType = NotFound;
    double bestSegmentSize = 0;
    double bestDistanceFromP1 = 0;

    int hopping = 0;

    if (firstPositionOnList < 0)
        firstPositionOnList = 0;

    for (int i = firstPositionOnList; i < lastPositionOnList; i++)
    {
        if (maxHopping > 0 && hopping > maxHopping)
            break;

        if (MapPose::areClose_2D(location, list[i]))
        {
            return NearestPoseSearchResult(i, Exact_P1, -1, -1);
        }
        if (MapPose::areClose_2D(location, list[i + 1]))
        {
            return NearestPoseSearchResult(i + 1, Exact_P2, -1, -1);
        }

        double distanceToSegment = MapPose::squaredDistanceToMidLine(list[i], list[i+1], location);

        if (distanceToSegment >= bestDistance) {
            hopping++;
            continue;
        }

        hopping = 0;

        // printf("projectOnPath: %f,%f, %f,%f\n", list[i]->x(),list[i]->y(),list[i+1]->x(), list[i+1]->y());
        auto res = MapPose::projectOnPath(list[i], list[i + 1], location);

        if (res == nullptr || res->pathSize == 0)
            continue;

        bestDistance = distanceToSegment;
        bestSegmentSize = res->pathSize;
        bestDistanceFromP1 = res->distanceFromP1;        

        // only the best cases are to be checked now

        if (MapPose::areClose_2D(res->pose, list[i]))
        {
            bestPos = i;
            bestPostType = Exact_P1;
            continue;
        }

        if (MapPose::areClose_2D(res->pose, list[i + 1]))
        {
            bestPos = i+1;
            bestPostType = Exact_P2;
            continue;
        }

        // BEFORE THE PATH SEGMENT
        if (res->distanceFromP1 < 0)
        {
            bestPos = i;
            bestPostType = Before_P1;
            continue;
        }
        // AFTER THE PATH SEGMENT
        if (res->distanceFromP1 > res->pathSize)
        {
            bestPos = i+1;
            bestPostType = After_P2;
            continue;
        }
        // INSIDE THE SEGMENT
        if (res->distanceFromP1 <= 0.5 * res->pathSize)
        {
            // p1 is the closest
            bestPos = i;
            bestPostType = After_P1;
            continue;
        }
        else
        {
            bestPos = i+1;
            bestPostType = Before_P2;
        }
    }

    return NearestPoseSearchResult(bestPos, bestPostType, bestSegmentSize, bestDistanceFromP1);
}

int MapPose::findBestNextPosInList(std::vector<MapPose> &list, MapPose &location, int firstPositionOnList, double too_close_distance, int maxHopping)
{

    NearestPoseSearchResult nearestRes = MapPose::findNearestPosePosOnList(list, location, firstPositionOnList, maxHopping);

    int lastPos = list.size() - 1;
    double distFromP2;

    switch (nearestRes.type)
    {
    case NearestPoseType::NotFound:
        return -2;

    case NearestPoseType::Before_P1:
        if (abs(nearestRes.bestDistanceFromP1) <= too_close_distance)
        {
            if (nearestRes.listPos == lastPos)
                return -1;
            return nearestRes.listPos + 1;
        }
        return nearestRes.listPos;
        break;
    case NearestPoseType::Before_P2:

        distFromP2 = nearestRes.bestSegmentSize - nearestRes.bestDistanceFromP1;
        if (distFromP2 <= too_close_distance)
        {
            if (nearestRes.listPos == lastPos)
                return -1;
            return nearestRes.listPos + 1;
        }
        return nearestRes.listPos;

        break;
    case NearestPoseType::Exact_P1:
    case NearestPoseType::Exact_P2:
        if (nearestRes.listPos == lastPos)
            return -1;
        return nearestRes.listPos + 1;
        break;
    case NearestPoseType::After_P1:
        if (nearestRes.bestSegmentSize <= too_close_distance)
        {
            // its after P1 but too close to P2
            if (nearestRes.listPos + 1 == lastPos)
                return -1;
            return nearestRes.listPos + 2;
        }
        // We dont need to check if we are at the last pos, because since it is P1, its safe to assume that the next pos is available
        return nearestRes.listPos + 1;
        break;
    case NearestPoseType::After_P2:
        if (nearestRes.listPos == lastPos)
            return -1;
        return nearestRes.listPos + 1;
        break;
    default:
        return -1;
    }
}

void MapPose::removeRepeatedSeqPointsInList(std::vector<MapPose> &list) {
    std::vector<int> rep;
    for (int i = 1; i < list.size(); i++) {
        if (list[i] == list[i-1])
            rep.push_back(i);
    }
    // Erase in reverse order to avoid index shifting
    for (int i = rep.size() - 1; i >= 0; --i) {
        list.erase(list.begin() + rep[i]);
    }   
}

extern "C"
{

    double map_pose_distance_to_line_2d(
        double line_x1,
        double line_y1,
        double line_x2,
        double line_y2,
        double x,
        double y)
    {
        return MapPose::distanceToLine_2D(line_x1, line_y1, line_x2, line_y2, x, y);
    }

    double map_pose_compute_path_heading_2d(
        double line_x1,
        double line_y1,
        double line_x2,
        double line_y2)
    {
        return MapPose::computePathHeading_2D_rad(line_x1, line_y1, line_x2, line_y2);
    }

    double *map_pose_project_on_path(
        double line_x1,
        double line_y1,
        double line_x2,
        double line_y2,
        double x,
        double y)
    {

        MapPose p1(line_x1, line_y1, 0.0, angle::rad(0.0));
        MapPose p2(line_x2, line_y2, 0.0, angle::rad(0.0));
        MapPose p(x, y, 0.0, angle::rad(0.0));

        std::unique_ptr<PathProjection> res = MapPose::projectOnPath(p1, p2, p);

        double *flattenRes = new double[6];
        flattenRes[0] = res->pose.x();
        flattenRes[1] = res->pose.y();
        flattenRes[2] = res->pose.z();
        flattenRes[3] = res->pose.heading().rad();
        flattenRes[4] = res->distanceFromP1;
        flattenRes[5] = res->pathSize;
        return flattenRes;
    }

    void map_pose_project_on_path_free(void *ptr)
    {
        double *p = (double *)ptr;
        delete[] p;
    }

    void *map_pose_store_pose_list(float *pose_list, int count)
    {
        std::vector<MapPose> *lst = new std::vector<MapPose>();
        lst->reserve(count);
        int pos = 0;
        for (int i = 0; i < count; i++)
        {
            pos = 4 * i;
            lst->push_back(MapPose(pose_list[pos], pose_list[pos + 1], pose_list[pos + 2], angle::rad(pose_list[pos + 3])));
        }
        return lst;
    }

    void map_pose_free_pose_list(void *list)
    {
        std::vector<MapPose> *lst = (std::vector<MapPose> *)list;
        lst->clear();
        delete lst;
    }

    int find_best_next_pose_on_list(void *list, double location_x, double location_y, int firstPositionOnList, double too_close_distance, int maxHopping)
    {
        std::vector<MapPose> *lst = (std::vector<MapPose> *)list;
        MapPose location(location_x, location_y, 0, angle::rad(0));
        return MapPose::findBestNextPosInList(*lst, location, firstPositionOnList, too_close_distance, maxHopping);
    }

    /*
        int *map_pose_find_nearest_pos_on_list_2d(void *list, double location_x, double location_y, int firstPositionOnList, double maxDistance)
        {
            std::vector<MapPose> *lst = (std::vector<MapPose> *)list;
            MapPose location(location_x, location_y, 0, angle::rad(0));
            auto res = MapPose::findNearestPosePosOnList(*lst, location, firstPositionOnList, maxDistance);
            return new int[2]{res.listPos, static_cast<int>(res.type)};
        }

        void map_pose_free_nearest_pos_on_list_result(void *res)
        {
            int *p = (int *)res;
            delete[] p;
        }*/
}