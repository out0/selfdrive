#pragma once

#ifndef __SEARCH_EGO_PARAMS_DRIVELESS_H
#define __SEARCH_EGO_PARAMS_DRIVELESS_H

#include <tuple>
#include <vector>
#include "angle.h"
#include "search_frame.h"
#include "math_utils.h"
#include "map_pose.h"
#include "world_pose.h"

class SearchParams
{
    friend class EgoParams;

    int _timeout_ms;
    float _maxPathSize_px;
    float _distanceToGoalTolerance_px;
    angle _headingErrorTolerance;
    std::pair<int, int> _minDistance;

    SearchFrame *_frame;
    Waypoint _start;
    Waypoint _goal;
    MapPose _ego_pose;
    MapPose _map_origin;
    WorldPose _world_origin;
    float _velocity_m_s;

    // Private constructor only accessible by Builder
    SearchParams(int timeout_ms, float maxPathSize_px, float distanceToGoalTolerance_px, angle headingErrorTolerance,
                 std::pair<int, int> minDistance, SearchFrame *frame, Waypoint start, Waypoint goal, MapPose ego_pose,
                 MapPose map_origin, WorldPose world_origin, float velocity_m_s) : _timeout_ms(timeout_ms), _maxPathSize_px(maxPathSize_px),
                                                                                   _distanceToGoalTolerance_px(distanceToGoalTolerance_px),
                                                                                   _headingErrorTolerance(headingErrorTolerance),
                                                                                   _minDistance(minDistance),
                                                                                   _frame(frame),
                                                                                   _start(start),
                                                                                   _goal(goal),
                                                                                   _ego_pose(ego_pose),
                                                                                   _map_origin(map_origin),
                                                                                   _world_origin(world_origin),
                                                                                   _velocity_m_s(velocity_m_s) {}

public:
    class SearchParamsBuilder
    {
        friend class SearchParams;

        int _timeout_ms = 350;
        float _maxPathSize_px = 30.0f;
        float _distanceToGoalTolerance_px = 20.0f;
        angle _headingErrorTolerance = angle::deg(5); // Assuming default angle constructor
        std::pair<int, int> _minDistance = {0, 0};

        SearchFrame *_frame = nullptr;
        Waypoint _start;
        Waypoint _goal;
        MapPose _ego_pose;
        MapPose _map_origin;
        WorldPose _world_origin;
        float _velocity_m_s = 1.0f;

        explicit SearchParamsBuilder(const Waypoint &start, const Waypoint &goal,
                                     const MapPose &ego_pose, const MapPose &map_origin, const WorldPose &world_origin) : _start(start), _goal(goal),
                                                                                                                          _ego_pose(ego_pose), _map_origin(map_origin), _world_origin(world_origin) {}

    public:
        SearchParamsBuilder &withTimeout(int timeout_ms)
        {
            _timeout_ms = timeout_ms;
            return *this;
        }

        SearchParamsBuilder &withMaxPathSize(float maxPathSize_px)
        {
            _maxPathSize_px = maxPathSize_px;
            return *this;
        }

        SearchParamsBuilder &withDistanceToGoalTolerance(float distanceToGoalTolerance_px)
        {
            _distanceToGoalTolerance_px = distanceToGoalTolerance_px;
            return *this;
        }

        SearchParamsBuilder &withHeadingErrorTolerance(angle headingErrorTolerance)
        {
            _headingErrorTolerance = headingErrorTolerance;
            return *this;
        }

        SearchParamsBuilder &withMinDistance(const std::pair<int, int> &minDistance)
        {
            _minDistance = minDistance;
            return *this;
        }

        SearchParamsBuilder &withFrame(SearchFrame *frame)
        {
            _frame = frame;
            return *this;
        }

        SearchParamsBuilder &withVelocity(float velocity_m_s)
        {
            _velocity_m_s = velocity_m_s;
            return *this;
        }

        SearchParamsBuilder &withMapOrigin(const MapPose &origin)
        {
            _map_origin = origin;
            return *this;
        }

        SearchParamsBuilder &withEgoPose(const MapPose &egoPose)
        {
            _ego_pose = egoPose;
            return *this;
        }

        SearchParamsBuilder &withWorldOrigin(const WorldPose &origin)
        {
            _world_origin = origin;
            return *this;
        }

        SearchParams build()
        {
            return SearchParams(_timeout_ms, _maxPathSize_px, _distanceToGoalTolerance_px, _headingErrorTolerance,
                                _minDistance, _frame, _start, _goal, _ego_pose, _map_origin, _world_origin, _velocity_m_s);
        }
    };

    static SearchParamsBuilder init(const Waypoint &start, const Waypoint &goal)
    {
        MapPose pose(0, 0, 0, angle::rad(0));
        MapPose map_origin(0, 0, 0, angle::rad(0));
        WorldPose origin(angle::rad(0), angle::rad(0), 0, angle::rad(0));
        return SearchParamsBuilder(start, goal, pose, map_origin, origin);
    }

    inline int timeout_ms() { return _timeout_ms; }
    inline float maxPathSize_px() { return _maxPathSize_px; }
    inline float distanceToGoalTolerance_px() { return _distanceToGoalTolerance_px; }
    inline angle headingErrorTolerance() { return _headingErrorTolerance; }
    inline std::pair<int, int> minDistance() { return _minDistance; }
    inline SearchFrame *frame() { return _frame; }
    inline Waypoint start() { return _start; }
    inline Waypoint goal() { return _goal; }

    inline MapPose ego_pose() { return _ego_pose; }
    inline MapPose map_origin() { return _map_origin; }
    inline WorldPose world_origin() { return _world_origin; }

    inline float velocity_m_s() { return _velocity_m_s; }
};

class EgoParams
{
private:
    std::tuple<int, int> _searchFrameDimensions;
    std::tuple<float, float> _searchFramePhysicalDimensions;
    std::vector<std::tuple<int, int, int>> _segmentationClassColors;
    std::vector<float> _segmentationClassCosts;

    std::pair<int, int> _egoLowerBound;
    std::pair<int, int> _egoUpperBound;

    angle _maxSteeringAngle;
    float _vehicleLength_m;
    float _maxCurvature;

    float _pixelToMeterRatio_Width;
    float _pixelToMeterRatio_Height;
    float _meterToPixelRatio_Width;
    float _meterToPixelRatio_Height;

    WorldPose _world_origin;

    EgoParams(
        const std::tuple<int, int> &searchFrameDimensions,
        const std::tuple<float, float> &searchFramePhysicalDimensions,
        const std::vector<std::tuple<int, int, int>> &segmentationClassColors,
        const std::vector<float> &segmentationClassCosts,
        const std::pair<int, int> &egoLowerBound,
        const std::pair<int, int> &egoUpperBound,
        angle maxSteeringAngle,
        float vehicleLength_m,
        float maxCurvature,
        float pixelToMeterRatio_Width,
        float pixelToMeterRatio_Height,
        float meterToPixelRatio_Width,
        float meterToPixelRatio_Height,
        const WorldPose &world_origin)
        : _searchFrameDimensions(searchFrameDimensions),
          _searchFramePhysicalDimensions(searchFramePhysicalDimensions),
          _segmentationClassColors(segmentationClassColors),
          _segmentationClassCosts(segmentationClassCosts),
          _egoLowerBound(egoLowerBound),
          _egoUpperBound(egoUpperBound),
          _maxSteeringAngle(maxSteeringAngle),
          _vehicleLength_m(vehicleLength_m),
          _maxCurvature(maxCurvature),
          _pixelToMeterRatio_Width(pixelToMeterRatio_Width),
          _pixelToMeterRatio_Height(pixelToMeterRatio_Height),
          _meterToPixelRatio_Width(meterToPixelRatio_Width),
          _meterToPixelRatio_Height(meterToPixelRatio_Height),
          _world_origin(world_origin)
    {
    }

public:
    class EgoParamsBuilder
    {
        friend class EgoParams;

        std::tuple<int, int> _searchFrameDimensions;
        std::tuple<float, float> _searchFramePhysicalDimensions = {-1, -1};
        std::vector<std::tuple<int, int, int>> _segmentationClassColors;
        std::vector<float> _segmentationClassCosts;

        std::pair<int, int> _egoLowerBound = {-1, -1};
        std::pair<int, int> _egoUpperBound = {-1, -1};

        angle _maxSteeringAngle = angle::deg(40);
        float _vehicleLength_m = 4.5f;
        float _maxCurvature = 0.35f;

        float _pixelToMeterRatio_Width = 1.0;
        float _pixelToMeterRatio_Height = 1.0;
        float _meterToPixelRatio_Width = 1.0;
        float _meterToPixelRatio_Height = 1.0;

        WorldPose _world_origin;

    public:
        explicit EgoParamsBuilder(std::tuple<int, int> searchFrameDimensions, WorldPose world_origin) : _searchFrameDimensions(searchFrameDimensions),
                                                                                                        _world_origin(world_origin) {}

        EgoParamsBuilder &withSearchPhysicalSize(float width_m, float height_m)
        {
            _searchFramePhysicalDimensions = {width_m, height_m};
            return *this;
        }

        EgoParamsBuilder &withSegmentationClassColors(const std::vector<std::tuple<int, int, int>> &colors)
        {
            _segmentationClassColors = colors;
            return *this;
        }

        EgoParamsBuilder &withSegmentationClassCosts(const std::vector<float> &costs)
        {
            _segmentationClassCosts = costs;
            return *this;
        }

        EgoParamsBuilder &withEgoLowerBound(const std::pair<int, int> &lowerBound)
        {
            _egoLowerBound = lowerBound;
            return *this;
        }

        EgoParamsBuilder &withEgoUpperBound(const std::pair<int, int> &upperBound)
        {
            _egoUpperBound = upperBound;
            return *this;
        }

        EgoParamsBuilder &withMaxSteeringAngle(angle maxSteeringAngle)
        {
            _maxSteeringAngle = maxSteeringAngle;
            return *this;
        }

        EgoParamsBuilder &withVehicleLength(float vehicleLength_m)
        {
            _vehicleLength_m = vehicleLength_m;
            return *this;
        }

        EgoParamsBuilder &withMaxCurvature(float maxCurvature)
        {
            _maxCurvature = maxCurvature;
            return *this;
        }

        EgoParamsBuilder &withWorldOrigin(WorldPose origin)
        {
            _world_origin = origin;
            return *this;
        }

        EgoParams build()
        {
            auto [xp, zp] = _searchFrameDimensions;
            auto [xm, zm] = _searchFramePhysicalDimensions;
            if (xm <= 0 || zm <= 0)
            {
                xm = 0.0f + xp;
                zm = 0.0f + zp;
                _searchFramePhysicalDimensions = {xm, zm};
            }

            _pixelToMeterRatio_Width = xm / xp;
            _pixelToMeterRatio_Height = zm / zp;
            _meterToPixelRatio_Width = xp / xm;
            _meterToPixelRatio_Height = zp / zm;

            return EgoParams(
                _searchFrameDimensions,
                _searchFramePhysicalDimensions,
                _segmentationClassColors,
                _segmentationClassCosts,
                _egoLowerBound,
                _egoUpperBound,
                _maxSteeringAngle,
                _vehicleLength_m,
                _maxCurvature,
                _pixelToMeterRatio_Width,
                _pixelToMeterRatio_Height,
                _meterToPixelRatio_Width,
                _meterToPixelRatio_Height,
                _world_origin);
        }
    };

    static EgoParamsBuilder init(const int searchWidth, const int searchHeight)
    {
        WorldPose origin(angle::rad(0), angle::rad(0), 0, angle::rad(0));
        return EgoParamsBuilder({searchWidth, searchHeight}, origin);
    }
    inline SearchParams::SearchParamsBuilder newSearchParams(const Waypoint &start, const Waypoint &goal)
    {
        return SearchParams::init(start, goal);
    }
    inline SearchParams::SearchParamsBuilder newSearchParams(const Waypoint &goal)
    {
        auto [w, h] = _searchFrameDimensions;
        return SearchParams::init(Waypoint(TO_INT(0.5 * w), TO_INT(0.5 * h), angle::rad(0)), goal);
    }

    SearchFrame *newSearchFrame();

    inline int width() { return std::get<0>(_searchFrameDimensions); }
    inline int height() { return std::get<1>(_searchFrameDimensions); }

    inline std::tuple<int, int> searchFrameDimensions() { return _searchFrameDimensions; }
    inline std::tuple<float, float> searchFramePhysicalDimensions() { return _searchFramePhysicalDimensions; }
    inline std::vector<std::tuple<int, int, int>> segmentationClassColors() { return _segmentationClassColors; }
    inline std::vector<float> segmentationClassCosts() { return _segmentationClassCosts; }

    inline std::pair<int, int> egoLowerBound() { return _egoLowerBound; }
    inline std::pair<int, int> egoUpperBound() { return _egoUpperBound; }

    inline angle maxSteeringAngle() { return _maxSteeringAngle; }
    inline float vehicleLength_m() { return _vehicleLength_m; }
    inline float maxCurvature() { return _maxCurvature; }

    inline float pixelToMeterRatio_Width() { return _pixelToMeterRatio_Width; }
    inline float pixelToMeterRatio_Height() { return _pixelToMeterRatio_Height; }
    inline float meterToPixelRatio_Width() { return _meterToPixelRatio_Width; }
    inline float meterToPixelRatio_Height() { return _meterToPixelRatio_Height; }

    inline WorldPose world_origin() { return _world_origin; }
};

#endif