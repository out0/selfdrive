#pragma once

#ifndef __SEARCH_FRAME_ZONE_UTILS_DRIVELESS_H
#define __SEARCH_FRAME_ZONE_UTILS_DRIVELESS_H

#include "cuda_basic.h"
#include "frame_params.h"

#define SEARCH_ZONE_POS(grid_width, xg, zg) (zg * grid_width + xg)
#define SEARCH_ZONE_TOTAL_OBSTACLES(sz_info, sz_pos) (sz_info[sz_pos].x)
#define SEARCH_ZONE_BORDER_OBSTACLES(sz_info, sz_pos) (sz_info[sz_pos].y)

#endif