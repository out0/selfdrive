#include <gtest/gtest.h>
#include "../../include/search_params.h"
#include "test_utils.h"
#include <cmath>
#include <tuple>

TEST(TestSearchParams, TestBuildEgoParamsAllDefault)
{
    EgoParams params = EgoParams::init(256, 256)
                           .build();

    auto v = std::pair<int, int>({-1, -1});
    ASSERT_EQ(params.egoLowerBound(), v);
    ASSERT_EQ(params.egoUpperBound(), v);

    ASSERT_EQ(params.maxSteeringAngle(), angle::deg(40));
    ASSERT_EQ(params.vehicleLength_m(), 4.5f);
    ASSERT_EQ(params.maxCurvature(), 0.35f);
    ASSERT_EQ(params.pixelToMeterRatio_Width(), 1.0f);
    ASSERT_EQ(params.pixelToMeterRatio_Height(), 1.0f);
    ASSERT_EQ(params.meterToPixelRatio_Width(), 1.0f);
    ASSERT_EQ(params.meterToPixelRatio_Height(), 1.0f);
}

TEST(TestSearchParams, TestBuildSearchParamsAllDefault)
{
    EgoParams ego = EgoParams::init(256, 256)
                        .build();

    SearchParams search = ego.newSearchParams(Waypoint(128, 0, angle::deg(11)))
                              .build();

    ASSERT_EQ(search.timeout_ms(), 350);
    ASSERT_EQ(search.maxPathSize_px(), 30.0f);
    ASSERT_EQ(search.distanceToGoalTolerance_px(), 20.0f);
    ASSERT_EQ(search.headingErrorTolerance(), angle::deg(5));
    ASSERT_EQ(search.minDistance(), (std::pair<int, int>(0, 0)));
    ASSERT_EQ(search.frame(), nullptr);
    ASSERT_TRUE(search.start() == Waypoint(128, 128, angle::deg(0)));
    ASSERT_TRUE(search.goal() == Waypoint(128, 0, angle::deg(11)));
    ASSERT_EQ(search.velocity_m_s(), 1.0f);
}

TEST(TestSearchParams, TestBuildEgoParamsCustomValues)
{
    EgoParams params = EgoParams::init(256, 256)
                           .withEgoLowerBound({10, 11})
                           .withEgoUpperBound({12, 13})
                           .withMaxCurvature(2.0)
                           .withMaxSteeringAngle(angle::deg(7))
                           .withSearchPhysicalSize(25.6, 25.6)
                           .withSegmentationClassColors({{0, 0, 0},
                                                         {255, 255, 255}})
                           .withSegmentationClassCosts({0.0,
                                                        -1.0})
                           .withVehicleLength(3.2)
                           .build();

    ASSERT_EQ(params.egoLowerBound(), (std::pair<int, int>({10, 11})));
    ASSERT_EQ(params.egoUpperBound(), (std::pair<int, int>({12, 13})));

    ASSERT_EQ(params.maxCurvature(), 2.0);
    ASSERT_EQ(params.maxSteeringAngle(), angle::deg(7));
    ASSERT_EQ(params.vehicleLength_m(), 3.2f);
    ASSERT_EQ(params.pixelToMeterRatio_Width(), 0.1f);
    ASSERT_EQ(params.pixelToMeterRatio_Height(), 0.1f);
    ASSERT_EQ(params.meterToPixelRatio_Width(), 10.0f);
    ASSERT_EQ(params.meterToPixelRatio_Height(), 10.0f);

    auto colors = params.segmentationClassColors();
    ASSERT_EQ(colors.at(0), (std::tuple<int, int, int>(0, 0, 0)));
    ASSERT_EQ(colors.at(1), (std::tuple<int, int, int>(255, 255, 255)));

    auto costs = params.segmentationClassCosts();
    ASSERT_EQ(costs.at(0), 0.0);
    ASSERT_EQ(costs.at(1), -1.0);
}

TEST(TestSearchParams, TestBuildSearchParamsCustomValues)
{
    EgoParams ego = EgoParams::init(256, 256)
                        .withEgoLowerBound({10, 11})
                        .withEgoUpperBound({12, 13})
                        .withMaxCurvature(2.0)
                        .withMaxSteeringAngle(angle::deg(7))
                        .withSearchPhysicalSize(25.6, 25.6)
                        .withSegmentationClassColors({{0, 0, 0},
                                                      {255, 255, 255}})
                        .withSegmentationClassCosts({0.0,
                                                     -1.0})
                        .withVehicleLength(3.2)
                        .build();

    SearchFrame *f = ego.newSearchFrame();

    SearchParams search = ego.newSearchParams(Waypoint(128, 107, angle::deg(-6.3)), Waypoint(128, 5, angle::deg(11)))
                              .withDistanceToGoalTolerance(10.12)
                              .withFrame(f)
                              .withHeadingErrorTolerance(angle::deg(12.3))
                              .withMaxPathSize(1.234)
                              .withMinDistance({10.12, 11.14})
                              .withTimeout(501)
                              .withVelocity(3.45)
                              .build();

    ASSERT_EQ(search.timeout_ms(), 501);
    ASSERT_EQ(search.maxPathSize_px(), 1.234f);
    ASSERT_EQ(search.distanceToGoalTolerance_px(), 10.12f);
    ASSERT_EQ(search.headingErrorTolerance(), angle::deg(12.3));
    ASSERT_EQ(search.minDistance(), (std::pair<int, int>(10.12, 11.14)));
    ASSERT_EQ(search.frame(), f);
    ASSERT_TRUE(search.start() == Waypoint(128, 107, angle::deg(-6.3)));
    ASSERT_TRUE(search.goal() == Waypoint(128, 5, angle::deg(11)));
    ASSERT_EQ(search.velocity_m_s(), 3.45f);
}