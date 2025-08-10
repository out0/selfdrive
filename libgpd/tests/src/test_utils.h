#ifndef __TEST_UTILS_DRIVELESS_H
#define __TEST_UTILS_DRIVELESS_H

#include <driveless/search_frame.h>

extern bool _ASSERT_DEQ(double a, double b, int tolerance = 4);
#define ASSERT_DEQ(a, b) ASSERT_TRUE(_ASSERT_DEQ(a, b))


void dump_search_frame_debug(SearchFrame &f1);

#endif