#include "../include/search_params.h"

SearchFrame * EgoParams::newSearchFrame()
{
    auto [w, h] = _searchFrameDimensions;
    SearchFrame *f = new SearchFrame(w, h, _egoLowerBound, _egoUpperBound);
    f->setClassCosts(_segmentationClassCosts);
    f->setClassColors(_segmentationClassColors);
    return f;
}