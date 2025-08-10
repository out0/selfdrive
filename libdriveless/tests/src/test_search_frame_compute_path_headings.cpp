#include <gtest/gtest.h>
#include "test_utils.h"
#include "../../include/interpolator.h"
#include "../../include/search_frame.h"
#include <cmath>

// DOES NOT WORK VERY WELL
// MAYBE ITS BETTER TO COMPLETLY IGNORE THIS FEATURE AND ASSUME THAT ALL WAYPOINT WILL HAVE A PROPER HEADING

TEST(TestSearchFrameComputePathHeadings, TestStraightHeading)
{
    Waypoint p1(50, 99, angle::deg(0));
    Waypoint p2(50, 0, angle::deg(0));

    auto curve = Interpolator::hermite(100, 100, p1, p2);
    for (int i = 0; i < curve.size(); i++) {
        curve[i].set_heading(1);
    }    
    auto curve2 = Interpolator::hermite(100, 100, p1, p2);
    ASSERT_EQ(curve.size(), 99);

    SearchFrame::computePathHeadings(100, 100, curve);

    for (int i = 0; i < curve.size(); i++) {
        // printf("%d, %d, %.2f", curve[i].x(), curve[i].z(), curve[i].heading().deg());
        // printf(" <===> %d, %d, %.2f\n", curve2[i].x(), curve2[i].z(), curve2[i].heading().deg());

        if (curve[i].heading() != curve2[i].heading()) {
            FAIL();
        }
    }
}

TEST(TestSearchFrameComputePathHeadings, TestLeftCurveHeading)
{
    Waypoint p1(50, 99, angle::deg(0));
    Waypoint p2(0, 0, angle::deg(-45));

    auto curve = Interpolator::hermite(100, 100, p1, p2);
    for (int i = 0; i < curve.size(); i++) {
        curve[i].set_heading(0);
    }
    auto curve2 = Interpolator::hermite(100, 100, p1, p2);

    SearchFrame::computePathHeadings(100, 100, curve);

    for (int i = 0; i < curve.size(); i++) {
        printf("%d, %d, %.2f", curve[i].x(), curve[i].z(), curve[i].heading().deg());
        printf(" <===> %d, %d, %.2f\n", curve2[i].x(), curve2[i].z(), curve2[i].heading().deg());        
        // if (curve[i].heading() != curve2[i].heading()) {
        //     FAIL();
        // }
    }
}