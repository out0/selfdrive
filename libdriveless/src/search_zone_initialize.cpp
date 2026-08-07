#include "../include/search_frame.h"

void SearchFrame::initialize_search_zones(int *params)
{
    if (_searchZoneDim.first <= 0 || _searchZoneDim.second <= 0)
    {
        _searchZoneDim.first = width() / 4;
        _searchZoneDim.second = height() / 4;
    }

    int W = TO_INT(round(static_cast<double>(width()) / static_cast<double>(_searchZoneDim.first)));
    int H = TO_INT(round(static_cast<double>(height()) / static_cast<double>(_searchZoneDim.second)));


    params[FRAME_SEARCH_ZONE_GRID_WIDTH] = W;
    params[FRAME_SEARCH_ZONE_GRID_HEIGHT] = W;
    params[FRAME_SEARCH_ZONE_DIM_WIDTH] = _searchZoneDim.first;
    params[FRAME_SEARCH_ZONE_DIM_HEIGHT] = _searchZoneDim.second;

    _search_zone_info = std::make_unique<Frame<uint4>>(W, H, _numCPUThreadHandlers);
}
