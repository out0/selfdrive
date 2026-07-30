#include "../include/search_frame.h"

void SearchFrame::initialize_search_zones()
{
    if (_searchZoneDim.first <= 0 || _searchZoneDim.second <= 0)
    {
        _searchZoneDim.first = width() / 4;
        _searchZoneDim.second = height() / 4;
    }

    int W = TO_INT(round(static_cast<double>(width()) / static_cast<double>(_searchZoneDim.first))) + 1;
    int H = TO_INT(round(static_cast<double>(height()) / static_cast<double>(_searchZoneDim.second))) + 1;

    _search_zone_info = std::make_unique<Frame<uint2>>(W, H, _numCPUThreadHandlers);
}
