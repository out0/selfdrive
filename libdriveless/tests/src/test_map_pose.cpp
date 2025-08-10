#include <gtest/gtest.h>
#include "../../include/map_pose.h"
#include "test_utils.h"
#include <cmath>

TEST(StateMapPose, TestSetGet)
{
    MapPose pose1(10.0, -10.2, 23.3456, angle::deg(10.2));
    ASSERT_FLOAT_EQ(pose1.x(), 10.0);
    ASSERT_FLOAT_EQ(pose1.y(), -10.2);
    ASSERT_FLOAT_EQ(pose1.z(), 23.3456);
    ASSERT_FLOAT_EQ(pose1.heading().deg(), 10.2);
}

TEST(StateMapPose, TestDistanceBetween)
{
    MapPose p1(0, 0, 0, angle::rad(0));
    MapPose p2(10, 0, 0, angle::rad(0));
    double dist = MapPose::DistanceBetween_2D(p1, p2);
    ASSERT_DEQ(dist, 10.0);

    MapPose p1b(0, 0, 0, angle::rad(0));
    MapPose p2b(10, 10, 0, angle::rad(0));
    dist = MapPose::DistanceBetween_2D(p1b, p2b);
    ASSERT_DEQ(dist, 10.0 * sqrt(2));

    MapPose p1c(-10, -10, 0, angle::rad(0));
    MapPose p2c(10, 10, 0, angle::rad(0));
    dist = MapPose::DistanceBetween_2D(p1c, p2c);
    ASSERT_DEQ(dist, 20.0 * sqrt(2));

    MapPose p1d(10, 10, 0, angle::rad(0));
    MapPose p2d(10, 10, 0, angle::rad(0));
    dist = MapPose::DistanceBetween_2D(p1d, p2d);
    ASSERT_DEQ(dist, 0.0);
}

TEST(StateMapPose, TestDistanceToLine)
{
    MapPose line_p1(2, 2, 0, angle::rad(0));
    MapPose line_p2(6, 6, 0, angle::rad(0));
    MapPose p(4, 0, 0, angle::rad(0));

    double dist = MapPose::distanceToLine_2D(line_p1, line_p2, p);
    ASSERT_DEQ(dist, 2 * sqrt(2));

    MapPose p2(6, 0, 0, angle::rad(0));
    dist = MapPose::distanceToLine_2D(line_p1, line_p2, p2);
    ASSERT_DEQ(dist, 3 * sqrt(2));
}

TEST(StateMapPose, TestPathHeading)
{
    MapPose line_p1(2, 2, 0, angle::rad(0));
    MapPose line_p2(6, 6, 0, angle::rad(0));

    angle heading = MapPose::computePathHeading_2D(line_p1, line_p2);
    ASSERT_TRUE(heading == angle::rad(PI / 4));

    heading = MapPose::computePathHeading_2D(line_p2, line_p1);
    ASSERT_TRUE(heading == angle::rad(-PI / 4 - PI / 2));

    MapPose line_p1_2(0, 0, 0, angle::rad(0));
    MapPose line_p2_2(0, 6, 0, angle::rad(0));

    heading = MapPose::computePathHeading_2D(line_p1_2, line_p2_2);
    ASSERT_TRUE(heading == angle::rad(PI / 2));

    heading = MapPose::computePathHeading_2D(line_p2_2, line_p1_2);
    ASSERT_TRUE(heading == angle::rad(-PI / 2));
}

TEST(StateMapPose, TestProjectOnPath)
{
    MapPose line_p1(2, 2, 0, angle::rad(0));
    MapPose line_p2(6, 6, 0, angle::rad(0));
    MapPose p(4, 0, 0, angle::rad(0));

    auto project = MapPose::projectOnPath(line_p1, line_p2, p);
    PathProjection *pathProjection = project.get();

    ASSERT_DEQ(project->distanceFromP1, 0.0);
    ASSERT_DEQ(project->pathSize, sqrt(32));
    ASSERT_TRUE(project->pose == MapPose(2, 2, 0, angle::rad(0)));

    MapPose p2(6, 0, 0, angle::rad(0));
    project = MapPose::projectOnPath(line_p1, line_p2, p2);

    ASSERT_DEQ(project->distanceFromP1, sqrt(2));
    ASSERT_DEQ(project->pathSize, sqrt(32));
    ASSERT_TRUE(project->pose == MapPose(3, 3, 0, angle::rad(0)));
}

TEST(StateMapPose, TestRemoveRepeatedSeqElements_x)
{
    std::vector<MapPose> list;

    for (int x = 0; x < 10; x++) {
        list.push_back(MapPose(x, 1, 1, angle::deg(0)));
        list.push_back(MapPose(x, 1, 1, angle::deg(0)));
    }

    ASSERT_EQ(20, list.size());
    MapPose::removeRepeatedSeqPointsInList(list);
    ASSERT_EQ(10, list.size());

    for (int i= 0; i < 10; i++) {
        float x = list[i].x();
        if (x != (float)i)
            FAIL();
    }

    // double remove should not affect the list
    MapPose::removeRepeatedSeqPointsInList(list);
    ASSERT_EQ(10, list.size());

    for (int i= 0; i < 10; i++) {
        if (list[i].x() != i)
            FAIL();
    }

    list.clear();
    for (int x = 0; x < 10; x++) {
        list.push_back(MapPose(x, 1, 1, angle::deg(0)));
    }
    for (int x = 0; x < 10; x++) {
        list.push_back(MapPose(x, 1, 1, angle::deg(0)));
    }
    // non sequential equal elements are not removed
    MapPose::removeRepeatedSeqPointsInList(list);
    ASSERT_EQ(20, list.size());
}
TEST(StateMapPose, TestRemoveRepeatedSeqElements_y)
{
    std::vector<MapPose> list;

    for (int y = 0; y < 10; y++) {
        list.push_back(MapPose(1, y, 1, angle::deg(0)));
        list.push_back(MapPose(1, y, 1, angle::deg(0)));
    }

    ASSERT_EQ(20, list.size());
    MapPose::removeRepeatedSeqPointsInList(list);
    ASSERT_EQ(10, list.size());

    for (int i= 0; i < 10; i++) {
        if (list[i].y() != (float)i)
            FAIL();
    }

    // double remove should not affect the list
    MapPose::removeRepeatedSeqPointsInList(list);
    ASSERT_EQ(10, list.size());

    for (int i= 0; i < 10; i++) {
        if (list[i].y() != i)
            FAIL();
    }

    list.clear();
    for (int y = 0; y < 10; y++) {
        list.push_back(MapPose(1, y, 1, angle::deg(0)));
    }
    for (int y = 0; y < 10; y++) {
        list.push_back(MapPose(1, y, 1, angle::deg(0)));
    }
    // non sequential equal elements are not removed
    MapPose::removeRepeatedSeqPointsInList(list);
    ASSERT_EQ(20, list.size());
}
TEST(StateMapPose, TestRemoveRepeatedSeqElements_z)
{
    std::vector<MapPose> list;

    for (int z = 0; z < 10; z++) {
        list.push_back(MapPose(1, 1, z, angle::deg(0)));
        list.push_back(MapPose(1, 1, z, angle::deg(0)));
    }

    ASSERT_EQ(20, list.size());
    MapPose::removeRepeatedSeqPointsInList(list);
    ASSERT_EQ(10, list.size());

    for (int i= 0; i < 10; i++) {
        if (list[i].z() != (float)i)
            FAIL();
    }

    // double remove should not affect the list
    MapPose::removeRepeatedSeqPointsInList(list);
    ASSERT_EQ(10, list.size());

    for (int i= 0; i < 10; i++) {
        if (list[i].z() != i)
            FAIL();
    }

    list.clear();
    for (int z = 0; z < 10; z++) {
        list.push_back(MapPose(1, 1, z, angle::deg(0)));
    }
    for (int z = 0; z < 10; z++) {
        list.push_back(MapPose(1, 1, z, angle::deg(0)));
    }
    // non sequential equal elements are not removed
    MapPose::removeRepeatedSeqPointsInList(list);
    ASSERT_EQ(20, list.size());
}
TEST(StateMapPose, TestRemoveRepeatedSeqElements_heading)
{
    std::vector<MapPose> list;

    for (int a = 0; a < 10; a++) {
        list.push_back(MapPose(1, 1, 1, angle::deg(a)));
        list.push_back(MapPose(1, 1, 1, angle::deg(a)));
    }

    ASSERT_EQ(20, list.size());
    MapPose::removeRepeatedSeqPointsInList(list);
    ASSERT_EQ(10, list.size());

    for (int i= 0; i < 10; i++) {
        if (list[i].heading() != angle::deg(i))
            FAIL();
    }

    // double remove should not affect the list
    MapPose::removeRepeatedSeqPointsInList(list);
    ASSERT_EQ(10, list.size());

    for (int i= 0; i < 10; i++) {
        if (list[i].heading() != angle::deg(i))
            FAIL();
    }

    list.clear();
    for (int a = 0; a < 10; a++) {
        list.push_back(MapPose(1, 1, 1, angle::deg(a)));
    }
    for (int a = 0; a < 10; a++) {
        list.push_back(MapPose(1, 1, 1, angle::deg(a)));
    }
    // non sequential equal elements are not removed
    MapPose::removeRepeatedSeqPointsInList(list);
    ASSERT_EQ(20, list.size());
}
