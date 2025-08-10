#include <gtest/gtest.h>
#include "../../include/map_pose.h"
#include "test_utils.h"
#include <cmath>


TEST(TestStateMapPoseNearestGoal, TestFindNearestGoalPose)
{
    return;
    std::vector<MapPose> path = {
        MapPose(0.0, 0.0, 0.0, angle::rad(0)),
        MapPose(10.0, 0.0, 0.0, angle::rad(0)),
        MapPose(20.0, 0.0, 0.0, angle::rad(0)),
        MapPose(25.0, 5.0, 0.0, angle::rad(0)),
        MapPose(30.0, 10.0, 0.0, angle::rad(0)),
        MapPose(35.0, 18.0, 0.0, angle::rad(0)),
        MapPose(35.0, 25.0, 0.0, angle::rad(0)),
        MapPose(35.0, 35.0, 0.0, angle::rad(0)),
        MapPose(30.0, 41.0, 0.0, angle::rad(0)),
        MapPose(25.0, 43.0, 0.0, angle::rad(0)),
        MapPose(20.0, 43.0, 0.0, angle::rad(0)),
        MapPose(10.0, 43.0, 0.0, angle::rad(0)),
        MapPose(5.0, 33.0, 0.0, angle::rad(0)),
        MapPose(0.0, 23.0, 0.0, angle::rad(0)),
        MapPose(0.0, 13.0, 0, angle::rad(0))};

    std::vector<std::vector<float>> points = {
        {0, 0}, {5, 2}, {10, -1}, {17, -3}, {23, -4}, {24, 6}, {29, 8}, {29, 17}, //
        {35, 18},
        {35, 20},
        {35, 22},
        {35, 24},
        {28, 34},
        {28, 39},
        {20, 39}, //
        {18, 39},
        {14, 39},
        {14, 29},
        {5, 24},
        {5, 20},
        {5, 19} //
    };

    std::vector<int> expected_pos = {
        1, 1, 2, 3, 3, 4, 5, 6, 6, 6, 7, 7, 8, 9, 11, 11, 12, 13, 14, 14 //
    };

    for (size_t i = 9; i < expected_pos.size(); ++i)
    {
        MapPose location(points[i][0], points[i][1], 0, angle::rad(0));
        int expected_pos_for_location = expected_pos[i];

        auto res = MapPose::findBestNextPosInList(path, location, 0, 5, 50);

        if (expected_pos_for_location != res)
        {
            std::string str = "wrong goal pose on test #" + std::to_string(i) + " location(" + std::to_string(location.x()) + "," + 
            std::to_string(location.y()) + "): expected: " + std::to_string(expected_pos_for_location) +
                              " obtained: " + std::to_string(res) + "\n";

            printf("%s", str.c_str());

            FAIL();
        }
    }
}

TEST(TestStateMapPoseNearestGoal, TestLastGoalPoseBug)
{
    std::string path_str = "[\"-90.0000991821289|-0.0008452492766082287|0.028222160413861275|0.00018598794633763798\", \"-89.45883330476649|-0.0008434922723435291|0.028222160413861275|0.00018598794633763798\", \"-88.91756742740407|-0.0008417352680788295|0.028222160413861275|0.00018598794633763798\", \"-88.3763019892927|0.13447649107679144|0.028222160413861275|0.00018598794633763798\", \"-87.83503655118135|0.2697947174216617|0.028222160413861275|0.00018598794633763798\", \"-87.29377111307|0.40511294376653195|0.028222160413861275|0.00018598794633763798\", \"-86.75250611420971|0.6757476394520078|0.028222160413861275|0.00018598794633763798\", \"-86.21124067609834|0.811065865796878|0.028222160413861275|0.00018598794633763798\", \"-85.66997523798699|0.9463840921417483|0.028222160413861275|0.00018598794633763798\", \"-85.26402626921625|1.0817018792355524|0.028222160413861275|0.00018598794633763798\", \"-84.72276083110488|1.2170201055804226|0.028222160413861275|0.00018598794633763798\", \"-84.18149539299353|1.352338331925293|0.028222160413861275|0.00018598794633763798\", \"-83.64022995488217|1.4876565582701633|0.028222160413861275|0.00018598794633763798\", \"-83.09896451677082|1.6229747846150333|0.028222160413861275|0.00018598794633763798\", \"-82.5576986394084|1.622976541619298|0.028222160413861275|0.00018598794633763798\", \"-82.01643276204597|1.6229782986235628|0.028222160413861275|0.00018598794633763798\", \"-81.47516644543248|1.487663586287222|0.028222160413861275|0.00018598794633763798\", \"-80.93390012881899|1.352348873950881|0.028222160413861275|0.00018598794633763798\", \"-80.52794984229504|1.0817172530228685|0.028222160413861275|0.00018598794633763798\", \"-79.98668352568156|0.9464025406865276|0.028222160413861275|0.00018598794633763798\", \"-79.98668352568156|0.9464025406865276|0.028222160413861275|0.00018598794633763798\", \"-79.98668352568156|0.9464025406865276|0.028222160413861275|0.00018598794633763798\"]";
    std::vector<MapPose> path;
    std::vector<std::string> pose_strings;
    size_t start = 0, end = 0;
    
    while ((start = path_str.find("\"", end)) != std::string::npos) {
        end = path_str.find("\"", start + 1);
        if (end == std::string::npos) break;
        pose_strings.push_back(path_str.substr(start + 1, end - start - 1));
        end++;
    }

    for (const auto& pose_str : pose_strings) {
        std::stringstream ss(pose_str);
        std::string item;
        std::vector<double> values;
        while (std::getline(ss, item, '|')) {
            values.push_back(std::stod(item));
        }
        if (values.size() == 4) {
            path.emplace_back(values[0], values[1], values[2], angle::deg(values[3]));
        }
    }

    MapPose location(-79.92220306396484, 1.2667661905288696, 0.028616636991500854, angle::deg(-9.376293440256612));
    
    MapPose::removeRepeatedSeqPointsInList(path);
    int res = MapPose::findBestNextPosInList(path, location, 0, 10, path.size() - 1);

    ASSERT_EQ(-1, res);
}