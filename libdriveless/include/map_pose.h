#pragma once

#ifndef __STATE_MAPPOSE_DRIVELESS_H
#define __STATE_MAPPOSE_DRIVELESS_H

#include "angle.h"

// CODE:BEGIN

#include <memory>
#include <tuple>
#include <vector>

class PathProjection;
class NearestPoseSearchResult;

class MapPose
{
    double _x;
    double _y;
    double _z;
    angle _heading;

    static inline double squaredDistanceToMidLine(MapPose &p1, MapPose &p2, MapPose &p);

    static NearestPoseSearchResult findNearestPosePosOnList(std::vector<MapPose> &list, MapPose &location, int firstPositionOnList, int maxHopping);

public:
    constexpr inline double x() const { return _x; }
    constexpr inline double y() const { return _y; }
    constexpr inline double z() const { return _z; }
    inline angle heading() { return _heading; } 

    MapPose(double x, double y, double z, angle heading) : _x(x), _y(y), _z(z), _heading(heading) {}

    inline bool operator==(const MapPose &other)
    {
        return __TOLERANCE_EQUALITY(_x, other._x) &&
               __TOLERANCE_EQUALITY(_y, other._y) &&
               __TOLERANCE_EQUALITY(_z, other._z) &&
               _heading == other._heading;
    }
    inline bool operator!=(const MapPose &other)
    {
        return !(*this == other);
    }

    inline MapPose clone();

    static inline bool areClose_2D(MapPose &p1, MapPose &p2);
    static inline bool areClose_3D(MapPose &p1, MapPose &p2);

    static double DistanceBetween_2D(MapPose &p1, MapPose &p2);
    static double DistanceBetween_3D(MapPose &p1, MapPose &p2);

    inline static double dot_2D(MapPose &p1, MapPose &p2);
    inline static double dot_3D(MapPose &p1, MapPose &p2);

    static double distanceToLine_2D(MapPose &lineP1, MapPose &lineP2, MapPose &p);
    static double distanceToLine_2D(
        double line_x1, 
        double line_y1, 
        double line_x2, 
        double line_y2,
        double x,
        double y);

    static angle computePathHeading_2D(MapPose &p1, MapPose &p2);
    static double computePathHeading_2D_rad(double line_x1, 
        double line_y1, 
        double line_x2, 
        double line_y2);

    static std::unique_ptr<PathProjection> projectOnPath(MapPose &p1, MapPose &p2, MapPose &p);

    static void removeRepeatedSeqPointsInList(std::vector<MapPose> &list);

    static int findBestNextPosInList(std::vector<MapPose> &list, MapPose &location, int firstPositionOnList, double too_close_distance, int maxHopping);
};

class PathProjection {
public:
    MapPose pose;
    double distanceFromP1;
    double pathSize;
};

typedef enum NearestPoseType {
    NotFound = 0,
    Before_P1 = 1,
    Exact_P1 = 2,
    After_P1 = 3,
    Before_P2 = 4,
    Exact_P2 = 5,
    After_P2 = 6
} NearestPoseType;

class NearestPoseSearchResult {
public:    
    int listPos;
    NearestPoseType type = NearestPoseType::NotFound;
    double bestSegmentSize;
    double bestDistanceFromP1;

    NearestPoseSearchResult(int _pos, NearestPoseType _type, double _bestSegmentSize, double _bestDistanceFromP1):
        listPos(_pos), type(_type), bestSegmentSize(_bestSegmentSize), bestDistanceFromP1(_bestDistanceFromP1) {}
};

// CODE:END

#endif