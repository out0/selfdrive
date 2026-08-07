#ifndef __TEST_UTILS_DRIVELESS_H
#define __TEST_UTILS_DRIVELESS_H

#include "../../include/search_frame.h"


extern bool _ASSERT_DEQ(double a, double b, int tolerance = 4);
#define ASSERT_DEQ(a, b) ASSERT_TRUE(_ASSERT_DEQ(a, b))



std::vector<Waypoint> testInterpolateHermiteCurve(int width, int height, Waypoint p1, Waypoint p2);

void exportSearchFrameToFile(SearchFrame &f, const char *file, bool show_zones = false, bool show_zone_edges = false);

void export_safe_distance_frame_minimal_dist_flag(SearchFrame &f, const char *file, std::vector<Waypoint> &path);

#endif